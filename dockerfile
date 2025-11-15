FROM agrigorev/zoomcamp-model:2025
WORKDIR /code
RUN pip install uv
COPY pyproject.toml .
COPY uv.lock .
COPY predict.py .
COPY final_model.pkl .


RUN uv sync

EXPOSE 9696

CMD ["uv", "run", "python", "inference.py"]
