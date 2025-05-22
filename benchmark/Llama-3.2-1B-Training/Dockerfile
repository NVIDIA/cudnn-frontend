FROM nvcr.io/nvidia/pytorch:25.04-py3
RUN pip install --upgrade pip && \
    pip install seaborn transformers
COPY training_perf.py /
WORKDIR /workspace
CMD ["python", "/training_perf.py"]
