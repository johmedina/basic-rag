export TOKENIZERS_PARALLELISM=false
echo "Starting Backend..."
uvicorn backend.main:app --reload &
sleep 2
echo "Starting Frontend..."
streamlit run frontend/app.py