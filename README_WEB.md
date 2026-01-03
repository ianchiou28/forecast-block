# Web Frontend for Forecast Block

This is a web frontend for the Forecast Block project, designed to mimic the "Finance Terminal" style.

## Prerequisites

- Python 3.8+
- Dependencies listed in `requirements_web.txt`

## How to Run

1.  Double-click `run_web.bat` (Windows).
2.  Or run manually:
    ```bash
    pip install -r requirements_web.txt
    python web_server.py
    ```
3.  Open your browser and navigate to `http://127.0.0.1:8000`.

## Features

-   **Daily Market Recap**: Displays the latest prediction report.
-   **Sentiment Analysis**: Visualizes market sentiment (based on limit-up data if available).
-   **Market Segmentation**: Shows market trends.
-   **Predictions List**: Lists the top predicted sectors.

## Customization

-   **Styles**: Edit `web/static/css/style.css` to change colors or layout.
-   **Templates**: Edit `web/templates/index.html` to change the HTML structure.
-   **Data Logic**: Edit `web_server.py` to change how data is fetched and processed.
