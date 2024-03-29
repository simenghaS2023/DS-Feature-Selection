# initialize python environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt 

# run the application
streamlit run main.py