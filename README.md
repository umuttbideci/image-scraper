# image-scraper
# AI Supported Image Scraper


- Every approved photo comes to output/approved.
- To edit clinic list edit clinics.txt (Every clinic from A to L has been done.)
- If you have any questions please text me from WhatsApp (+90 545 766 6020)
  
# First, set the API Key

  $ echo 'export SERPAPI_KEY="YOUR API KEY HERE"' >> ~/.zshrc
  $ source ./.zshrc

# Run the script (--clinic-workers x, x clinics fetched from the google images at the same time.)
  
  $ python single_clinic.py --clinic-workers 3  # Process 3 clinics at once   

