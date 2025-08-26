# 🏥 AI-Powered Hair Transplant Clinic Image Scraper

An intelligent image scraper that automatically finds and filters high-quality photos of hair transplant clinics using AI (CLIP model) and Google Images API.

## ✨ Features

- **AI-Powered Filtering**: Uses CLIP model to identify professional clinic photos
- **Parallel Processing**: Process multiple clinics simultaneously (default: 5 at once)
- **Smart Image Selection**: Automatically filters out low-quality, duplicate, and irrelevant images
- **Batch Processing**: Process entire clinic lists from `clinics.txt`
- **Quality Metrics**: Blur detection, size validation, and content scoring

## 🚀 Quick Start (No Setup Required!)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd image-scraper
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Your SerpAPI Key
- Go to [SerpAPI](https://serpapi.com/) and sign up
- Get your free API key from the dashboard
- Copy the key (looks like: `2ddcf26be035d7222107f0b9e1adde283a757869317385b45585947112529186`)

### 4. Set API Key (Choose One Method)

#### Option A: Environment Variable (Recommended)
```bash
export SERPAPI_KEY="your_api_key_here"
```

#### Option B: Direct Parameter
```bash
python3 single_clinic.py --serpapi-key "your_api_key_here" --clinics-file clinics.txt
```

#### Option C: Permanent Setup (macOS/Linux)
```bash
echo 'export SERPAPI_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### 5. Run the Script
```bash
python3 single_clinic.py --clinics-file clinics.txt
```

## 📁 File Structure

```
image-scraper/
├── single_clinic.py          # Main script
├── clinics.txt               # List of clinics to process
├── requirements.txt           # Python dependencies
├── output/                   # Generated results
│   ├── approved/            # Approved clinic photos
│   └── report_*.csv         # Detailed reports per clinic
└── README.md                # This file
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Process all clinics with default settings
python3 single_clinic.py --clinics-file clinics.txt
```

### Advanced Usage
```bash
# Process 3 clinics simultaneously, 5 approved photos per clinic
python3 single_clinic.py \
  --clinics-file clinics.txt \
  --target-approved 5 \
  --clinic-workers 3 \
  --max-pages 10 \
  --debug
```

### Custom Settings
```bash
python3 single_clinic.py \
  --clinics-file clinics.txt \
  --target-approved 3 \
  --max-pages 15 \
  --threads 8 \
  --clinic-workers 5 \
  --min-side 800 \
  --blur-min 150.0 \
  --clip-pos-min 0.25 \
  --clip-margin-min 0.15
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--clinics-file` | `clinics.txt` | File containing clinic names (one per line) |
| `--target-approved` | `3` | Number of approved photos per clinic |
| `--max-pages` | `5` | Maximum Google Images pages to search |
| `--clinic-workers` | `5` | Number of clinics to process simultaneously |
| `--threads` | `12` | Number of concurrent image downloads |
| `--min-side` | `600` | Minimum image dimension (width/height) |
| `--blur-min` | `100.0` | Minimum blur score (higher = sharper) |
| `--clip-pos-min` | `0.20` | Minimum positive CLIP score |
| `--clip-margin-min` | `0.12` | Minimum CLIP score margin |
| `--no-suffix` | `False` | Don't add " klinik" suffix to queries |
| `--debug` | `False` | Enable debug logging |

## 🔧 Requirements

- Python 3.8+
- Internet connection
- SerpAPI account and key
- 4GB+ RAM (for AI models)

## 📊 Output

The script generates:
- **Approved Photos**: High-quality clinic images in `output/approved/<clinic_name>/`
- **Individual Reports**: `output/report_<clinic_name>.csv` for each clinic
- **Summary Report**: `output/report_all.csv` with all clinics' results

## 🚨 Troubleshooting

### Common Issues

**"SerpAPI key yok" Error**
```bash
# Make sure you've set the API key
export SERPAPI_KEY="your_key_here"
echo $SERPAPI_KEY  # Should show your key
```

**"clinics.txt bulunamadı" Error**
```bash
# Create clinics.txt with clinic names
echo "Dr. Ahmet Klinik" > clinics.txt
echo "Esteworld Klinik" >> clinics.txt
```

**Memory Issues**
```bash
# Reduce parallel processing
python3 single_clinic.py --clinic-workers 2 --threads 6
```

**Rate Limiting**
```bash
# Reduce concurrent requests
python3 single_clinic.py --threads 4
```

## 📝 Customization

### Edit Clinic List
```bash
# Edit clinics.txt
nano clinics.txt

# Add your clinics (one per line):
Dr. Mehmet Klinik
Esteworld Klinik
Dr. Ahmet Klinik
```

### Adjust AI Thresholds
```bash
# More strict filtering
python3 single_clinic.py --clip-pos-min 0.30 --clip-margin-min 0.20

# More lenient filtering
python3 single_clinic.py --clip-pos-min 0.15 --clip-margin-min 0.08
```

## 🤝 Support

- **WhatsApp**: +90 545 766 6020
- **Issues**: Create an issue in this repository
- **Documentation**: Check the code comments for detailed explanations

## 📄 License

This project is for educational and research purposes. Please respect SerpAPI's terms of service and rate limits.

---

**Happy Scraping! 🎉**

