# Abuse-Ring Sentinel — Live Dashboard

This is a real, working demo: a FastAPI backend trains your actual HMM
(`hmm/model_v3.py`) and Bayesian fusion (`bayesian/risk_model.py`) logic
at startup, and a browser dashboard calls it live over HTTP. Nothing in
the dashboard is hardcoded — every number comes from the trained model
responding to a real API request.

## What's in here

```
dashboard/
  backend/
    main.py             FastAPI app — /health, /model-info, /score
    model_service.py     Your HMM + Bayesian logic, refactored into a
                          reusable service instead of a one-shot script
    requirements.txt
  frontend/
    index.html            The dashboard UI (matches your site's design)
  data/
    behavioral_sequences_v2.csv   Your preprocessed dataset (needed to train)
```

## 1. Install & run the backend

```bash
cd dashboard/backend
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Watch the terminal — on startup it trains the HMM (same 80/20
entity-level split, same GaussianHMM config as your evaluation run).
This takes anywhere from a few seconds to about a minute depending on
your machine. You'll see:

```
Training HMM + fitting Bayesian evidence — this runs once...
Model ready.
{'train_entities': 11876, 'test_entities': 2969, ...}
```

Those numbers should match your `final_model_comparison.csv` split
exactly (11,876 train entities / 2,969 test entities / 52,902 test
observations) — that's your proof the API is running the real
pipeline, not a mock.

## 2. Open the dashboard

Just open `frontend/index.html` directly in your browser (double-click
it, or right-click → Open With → your browser). It talks to
`http://localhost:8000` by default.

If you want to serve it instead of opening the file directly:
```bash
cd dashboard/frontend
python3 -m http.server 5500
```
then visit `http://localhost:5500`.

## 3. For the video

1. Show the terminal training on startup — real console output, real
   numbers, proves it's not faked.
2. Open the dashboard, point out the "Connected — model trained and
   ready" status bar and the stats strip (train/test entities, fraud
   prior, HMM converged) — these are read live from `/health`, not
   written into the HTML.
3. Build a session: add a window with everyday-looking values (few
   transactions, 1 device, no rapid-fire) — show it lands NORMAL /
   ALLOW.
4. Add 2–3 more windows escalating (more devices, more payment
   emails, rapid-fire checked) — show the state flip to PROBING then
   ACTIVE_ABUSE, the trajectory score and fused risk probability climb,
   and the decision flip to REVIEW then BLOCK, with the "Why" column
   explaining exactly which evidence fired.
5. Mention in voiceover: "this is the same HMM and Bayesian fusion
   code from the repo, running live behind an API — the dashboard
   isn't showing you canned numbers."

## Notes / honesty for Q&A

- A session started in the dashboard has no real prior history, so
  trajectory fields that depend on "what happened before" (probing
  history, recent-active count, persistence) build up fresh from only
  the windows you add in that session — same formula as
  `hmm/model_v3.py`, just starting from a clean slate, the way a
  brand-new account would.
- Fields not exposed as sliders (transaction amounts, receiver emails,
  product count) use sensible dataset-informed defaults so the model
  still gets a complete feature vector — you can see/change these in
  `model_service.py`'s `DEFAULT_WINDOW` if you want to expose more
  sliders later.
- The Bayesian evidence likelihoods (`P(evidence | fraud)` etc.) are
  estimated once at startup from the same held-out test set your
  evaluation used — they're fixed for the life of the running server,
  not recalculated per request.
