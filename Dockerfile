# FROM python:3.8

# # Install Poetry
# RUN pip install --upgrade pip && \
#     pip install poetry

# # Copy project files and install dependencies
# WORKDIR /app
# COPY pyproject.toml poetry.lock ./
# RUN poetry config virtualenvs.create false 
# RUN poetry install --no-interaction --no-ansi --no-root

# # Copy only the necessary files for application execution
# COPY . .
# # COPY utils/ ./utils/

# # Set environment variables
# ENV PYTHONPATH=/app
# EXPOSE 8500


# ENTRYPOINT ["streamlit", "run", "--server.headless", "true", \
#             "--server.port", "8500", "stream_test.py"]

FROM python:3.8

# Install Poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.1.8 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="$POETRY_HOME/bin:$PATH" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install CUDA-enabled PyTorch if available, fall back to CPU-only version otherwise
RUN if [ $(dpkg-query -W -f='${Status}' nvidia-cuda-toolkit 2>/dev/null | grep -c "ok installed") -eq 1 ]; then \
        echo "CUDA detected, installing CUDA-enabled PyTorch"; \
        pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html; \
    else \
        echo "No CUDA detected, installing CPU-only PyTorch"; \
        pip install torch torchvision torchaudio; \
    fi

# Copy project files and install dependencies
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false 
RUN poetry install --no-interaction --no-ansi --no-root

# Copy only the necessary files for application execution
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
EXPOSE 8500

ENTRYPOINT ["streamlit", "run", "--server.headless", "true", \
            "--server.port", "8500", "stream_test.py"]

