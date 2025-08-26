#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, sys, time, hashlib, argparse, re
from pathlib import Path
import logging
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import cv2
import pandas as pd
import imagehash
from tqdm import tqdm
from serpapi import GoogleSearch

import torch
from transformers import AutoProcessor, CLIPModel

# ================== Prompts ==================
PROMPT_POS = "a clean, professional interior or outdoor photo of (might include patients) a hair transplant clinic, medical environment, bright lighting"
PROMPT_NEG = "selfie, messy background, watermark, logo, collage, cartoon, low quality, non-medical, crowd"

# ================== Logging ==================
def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

# ================== Utils ==================
def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:64] if s else "query"

def device_pick() -> str:
    # Kullanıcı CPU istedi; GPU kapalı
    return "cpu"

# ================== Models ==================
def load_models(device: str, logger):
    logger.info("CLIP modeli (CPU) yükleniyor…")
    t0 = time.time()
    clip_model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_id)
    clip_proc  = AutoProcessor.from_pretrained(clip_model_id)
    clip_model.to(device).eval()
    logger.info("Model hazır (%.1fs).", time.time() - t0)
    return clip_model, clip_proc

# ================== SerpAPI ==================
def google_images_page(query: str, api_key: str, ijn: int, safe="active"):
    """
    SerpAPI'den Google görsellerini getirir.
    Önce engine=google_images dener; boşsa engine=google + tbm=isch fallback.
    """
    log = getLogger("serpapi")

    def _extract_urls(res: dict):
        items = res.get("images_results") or res.get("image_results") or []
        urls = []
        for it in items:
            u = it.get("original") or it.get("link") or it.get("thumbnail")
            if u: urls.append(u)
        return urls, len(items)

    # 1) Tercih: engine=google_images
    params = {
        "engine": "google_images",
        "q": query,
        "ijn": str(ijn),
        "api_key": api_key,
        "no_cache": "true",
        "device": "desktop",
        "hl": "tr",
        # "tbs": "itp:photos,isz:m",  # sadece foto/medium (istersen aç)
    }
    res = GoogleSearch(params).get_dict()
    if "error" in res:
        log.error("SerpAPI error(engine=google_images): %s", res["error"])
    urls, count = _extract_urls(res)
    log.debug("google_images ijn=%s items=%d urls=%d", ijn, count, len(urls))
    if urls:
        return urls

    # 2) Fallback: engine=google + tbm=isch
    params_fb = {
        "engine": "google",
        "q": query,
        "tbm": "isch",
        "ijn": str(ijn),
        "api_key": api_key,
        "no_cache": "true",
        "safe": safe,
        "hl": "tr",
    }
    res_fb = GoogleSearch(params_fb).get_dict()
    if "error" in res_fb:
        log.error("SerpAPI error(engine=google,tbm=isch): %s", res_fb["error"])
    urls_fb, count_fb = _extract_urls(res_fb)
    log.debug("google+tbm=isch ijn=%s items=%d urls=%d", ijn, count_fb, len(urls_fb))
    return urls_fb

# ================== Image helpers ==================
def download_image(url: str, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return url, Image.open(io.BytesIO(r.content)).convert("RGB"), None
    except Exception as e:
        return url, None, str(e)

def blur_score(img: Image.Image) -> float:
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def phash_str(img: Image.Image) -> str:
    return str(imagehash.phash(img))

@torch.no_grad()
def clip_scores(img: Image.Image, processor, model, device: str):
    inputs = processor(text=[PROMPT_POS, PROMPT_NEG], images=img, return_tensors="pt", padding=True).to(device)
    out = model(**inputs)
    ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    sims = (ie @ te.T).squeeze(0).detach().cpu().tolist()
    pos, neg = float(sims[0]), float(sims[1])
    return pos, neg, pos - neg

# ================== Per-clinic ==================
def process_one_clinic(base_query: str, args, device: str, clip_model, clip_proc, logger):
    # Sorguya otomatik " klinik" ekle (istenirse kapatılabilir)
    query = base_query if (args.no_suffix or base_query.endswith(" klinik")) else f"{base_query} klinik"

    qslug = slugify(query)
    root = Path(args.out)
    app_dir = root / "approved" / qslug
    rep_csv = root / f"report_{qslug}.csv"
    app_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    phashes_seen = set()
    url_seen = set()
    approved = rejected = errors = 0

    logger.info("[%-20s] hedef=%d, max_pages=%d, threads=%d (CPU)",
                qslug[:20], args.target_approved, args.max_pages, args.threads)

    for page in range(args.max_pages):
        logger.info("Sayfa %d (ijn=%d) çekiliyor…", page+1, page)
        urls = google_images_page(query, args.serpapi_key, page)
        urls = [u for u in urls if u and u not in url_seen]
        url_seen.update(urls)
        if not urls:
            logger.warning("Bu sayfada URL çıkmadı.")
            continue

        # İndirme: I/O-bound → threads
        futs = []
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            for u in urls:
                futs.append(ex.submit(download_image, u))

            for f in tqdm(as_completed(futs), total=len(futs),
                          desc=f"[{qslug}] Sayfa {page+1}: indir/puanla", unit="img", leave=False):
                url, img, err = f.result()
                if img is None:
                    errors += 1
                    rows.append({"url": url, "status": "error", "reason": f"download: {err}"})
                    continue

                # duplicate kontrol
                try:
                    ph = phash_str(img)
                    if ph in phashes_seen:
                        rows.append({"url": url, "status": "duplicate", "reason": "pHash"})
                        rejected += 1
                        continue
                    phashes_seen.add(ph)
                except Exception as e:
                    errors += 1
                    rows.append({"url": url, "status": "error", "reason": f"pHash: {e}"})
                    continue

                # boyut & blur eşikleri (min-side 600)
                w, h = img.size
                if (w < args.min_side) or (h < args.min_side):
                    rows.append({"url": url, "status": "rejected", "reason": f"too_small {w}x{h}"})
                    rejected += 1
                    continue

                b = blur_score(img)
                if b < args.blur_min:
                    rows.append({"url": url, "status": "rejected", "reason": f"blur={b:.1f} < {args.blur_min}"})
                    rejected += 1
                    continue

                # CLIP puanlama
                try:
                    pos, neg, margin = clip_scores(img, clip_proc, clip_model, device)
                except Exception as e:
                    errors += 1
                    rows.append({"url": url, "status": "error", "reason": f"clip: {e}"})
                    continue

                ok = (pos >= args.clip_pos_min) and (margin >= args.clip_margin_min)
                fname = f"{hashlib.md5(url.encode()).hexdigest()[:10]}.jpg"
                if ok:
                    img.save(app_dir / fname, "JPEG", quality=92)
                    rows.append({"url": url, "status": "approved",
                                 "pos": pos, "neg": neg, "margin": margin, "blur": b,
                                 "save": str(app_dir / fname)})
                    approved += 1
                else:
                    rows.append({"url": url, "status": "rejected",
                                 "pos": pos, "neg": neg, "margin": margin, "blur": b,
                                 "reason": "clip_threshold"})
                    rejected += 1

                if approved >= args.target_approved:
                    break  # bu sayfayı ve kliniği bitir

        if approved >= args.target_approved:
            logger.info("Hedefe ulaşıldı: %d onay.", approved)
            break

    # rapor
    root.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(rep_csv, index=False)
        logger.info("Rapor yazıldı: %s", rep_csv)
    return {"query": query, "approved": approved, "rejected": rejected, "errors": errors,
            "approved_dir": str(app_dir.resolve()), "report": str(rep_csv)}

# ================== Main ==================
def main():
    ap = argparse.ArgumentParser(description="clinics.txt içinden klinikleri PARALEL işler; klinik başına 3 onaylı foto bulunca durur.")
    ap.add_argument("--clinics-file", default="clinics.txt", help="Her satır bir klinik adı.")
    ap.add_argument("--serpapi-key", default=os.getenv("SERPAPI_KEY", ""), help="SerpAPI key (ya da env).")
    ap.add_argument("--out", default="output", help="Çıkış kök klasörü.")

    # Varsayılanlar: hedef 3, en fazla 5 sayfa
    ap.add_argument("--target-approved", type=int, default=3, help="Klinik başına hedef onay sayısı (default: 3)")
    ap.add_argument("--max-pages", type=int, default=5, help="Klinik başına en fazla sayfa (default: 5)")
    ap.add_argument("--threads", type=int, default=12, help="Aynı anda kaç indirme (I/O)")
    ap.add_argument("--clinic-workers", type=int, default=5, help="Aynı anda kaç klinik işlensin (default: 5)")

    # Filtre eşikleri
    ap.add_argument("--min-side", type=int, default=600)
    ap.add_argument("--blur-min", type=float, default=100.0)
    ap.add_argument("--clip-pos-min", type=float, default=0.20)
    ap.add_argument("--clip-margin-min", type=float, default=0.12)

    ap.add_argument("--no-suffix", action="store_true", help="Sorguların sonuna ' klinik' ekleme")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
   

    setup_logging(args.debug)
    logger = getLogger("clinics_seq")

    if not args.serpapi_key:
        logger.error("SerpAPI key yok. --serpapi-key veya SERPAPI_KEY env ver.")
        sys.exit(2)

    device = device_pick()
    logger.info("Cihaz: %s (CPU)", device.upper())
    clip_model, clip_proc = load_models(device, logger)

    # clinics.txt oku
    path = Path(args.clinics_file)
    if not path.exists():
        logger.error("Klinik listesi bulunamadı: %s", args.clinics_file); sys.exit(2)
    clinics = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Paralel işlem (configurable klinik aynı anda)
    results = []
    max_clinic_workers = args.clinic_workers
    
    with ThreadPoolExecutor(max_workers=max_clinic_workers) as clinic_executor:
        # Submit all clinic processing jobs
        future_to_clinic = {
            clinic_executor.submit(process_one_clinic, base, args, device, clip_model, clip_proc, logger): base
            for base in clinics
        }
        
        with tqdm(total=len(clinics), desc=f"Clinics (parallel x{max_clinic_workers})", unit="clinic", dynamic_ncols=True) as cbar:
            for future in as_completed(future_to_clinic):
                clinic_name = future_to_clinic[future]
                try:
                    res = future.result()
                    results.append(res)
                    logger.info("=== Tamamlandı: %s (approved=%d) ===", clinic_name, res['approved'])
                except Exception as exc:
                    logger.error("Klinik %s işlenirken hata: %s", clinic_name, exc)
                    results.append({
                        "query": clinic_name, "approved": 0, "rejected": 0, "errors": 1,
                        "approved_dir": "", "report": "", "error": str(exc)
                    })
                finally:
                    cbar.update(1)
                    remaining = cbar.total - cbar.n
                    cbar.set_postfix_str(f"remaining={remaining}")

    # Özet CSV
    summary_csv = out_dir / "report_all.csv"
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    logger.info("Tüm kliniklerin özeti: %s", summary_csv)

if __name__ == "__main__":
    main()
