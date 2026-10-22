# CRANE-LLM demo: JupyterLab with CRANE-LLM and the demo notebook, ready to run.
#
# Build, from this directory:
#   docker build -t crane-llm-demo .
# Run, then open http://localhost:8888:
#   docker run --rm -p 127.0.0.1:8888:8888 -e CRANE_LLM_API_KEY=sk-... crane-llm-demo
#
# Every version is pinned, so the image behaves the same however long after
# the release it is built. Bump CRANE_LLM_VERSION for a new release.

FROM python:3.11-slim

ARG CRANE_LLM_VERSION=1.0.1

LABEL org.opencontainers.image.title="CRANE-LLM demo" \
      org.opencontainers.image.description="JupyterLab with CRANE-LLM and its demo notebook" \
      org.opencontainers.image.source="https://github.com/yarinamomo/crane_llm" \
      org.opencontainers.image.licenses="BSD-3-Clause"

RUN pip install --no-cache-dir \
        "crane-llm==${CRANE_LLM_VERSION}" \
        jupyterlab==4.5.9 \
        ipykernel==6.29.5 \
        numpy==2.2.6 \
        pandas==2.2.3 \
        scikit-learn==1.6.1 \
        matplotlib==3.10.3 \
        seaborn==0.13.2

# Run as an ordinary user. ~/.crane_llm/config.json, written by set_api_key,
# lives in this user's home.
RUN useradd --create-home --uid 1000 crane
USER crane
WORKDIR /home/crane/cranellm_demo
COPY --chown=crane:crane cranellm_demo.ipynb README.md ./
COPY --chown=crane:crane data/data_test.csv data/

EXPOSE 8888

# No token: the server is meant to be reached from this machine only, which is
# why the run command above publishes the port on 127.0.0.1.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--LabApp.default_url=/lab/tree/cranellm_demo.ipynb"]
