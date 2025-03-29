# BIHAR-2025

## Project Structure
```
BIHAR-2025/
├── notebooks/
├── data/
│   ├── multimodal/
│   └── timeseries/
├── model/
│   ├── image_model/
│   ├── text_model/
│   ├── multimodal_model/
│   ├── timeseries_model/
│   └── registry/
├── monitoring/
│   ├── generate_comparison.py
│   └── output/
├── api/
└── README.md
```

## API Endpoints
- `POST /models/{type}/predict`
- `GET /version`


## Data Acquisition

### 

Place the json file from Kaggle API token under /config

### Installation & Setup
```bash
# Install required packages
pip install kaggle python-dotenv

# Execute from project root directory
python data/fetch_data.py