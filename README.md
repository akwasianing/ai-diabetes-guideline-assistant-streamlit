
# AI Clinical Decision Support Assistant for Type 2 Diabetes

This Streamlit application implements an AI-enhanced Retrieval-Augmented Generation
pipeline for querying selected ADA Standards of Care in Diabetes—2026 documents.

## AI pipeline

- Semantic embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Initial retrieval: cosine similarity
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Answer generation: Groq-hosted `openai/gpt-oss-20b`
- Interface: Streamlit
- Grounding: source labels tied to PDF file names and page numbers

## Configure the Groq API key

Create a free Groq API key, then create this local file:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "paste-your-key-here"
```

Never upload `secrets.toml` to GitHub.

For Streamlit Community Cloud:

1. Open the app settings.
2. Select **Secrets**.
3. Add the same TOML line.
4. Save and reboot the app.

## Run on macOS

Open Terminal in this project directory and run:

```bash
chmod +x run_mac.command
./run_mac.command
```

The first launch may take several minutes while the semantic models download and
the local embedding index is built.

## Run manually

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Important safety and privacy limitations

- Educational and portfolio prototype only.
- Not a medical device and not HIPAA compliant.
- Do not enter identifiable patient information.
- Does not provide patient-specific diagnosis or treatment.
- Generated answers must be checked against the cited source pages.
- The language model receives the question and retrieved passages through the Groq API.

## Copyright and deployment notice

The included ADA documents were supplied for the owner's educational project.
The documents themselves contain restrictions relating to reproduction,
third-party hosting, and text/data mining. Review the publisher's current permission
terms before putting the documents or a document-backed public application online.
A public portfolio version should use content for which public hosting and machine
processing permission has been confirmed.
