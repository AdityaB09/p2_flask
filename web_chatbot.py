import numpy as np
from flask import Flask, request, jsonify, render_template

from project_p2_light import (
    load_as_list,
    load_glove,
    EMBEDDING_FILE,
    instantiate_models,
    train_model_glove,
    string2vec,
    get_dependency_parse,
    get_dep_categories,
    custom_feature,
)

# ------------- MODEL SETUP (runs once at startup) -------------

print("Loading dataset and GloVe...")
documents, labels = load_as_list("dataset.csv")
glove_reps = load_glove(EMBEDDING_FILE)

print("Instantiating models...")
nb_glove, logistic_glove, svm_glove, mlp_glove = instantiate_models()

print("Training SVM (GloVe)... this may take a bit on first run.")
svm_glove = train_model_glove(svm_glove, glove_reps, documents, labels)
model = svm_glove

print("Model loaded and ready.")


# ------------- FLASK APP -------------

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/", methods=["GET"])
def index():
    # Render the chat UI
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    JSON input:  { "message": "I feel happy today" }
    JSON output: {
        "sentiment": "...",
        "style": "..."
    }
    """
    data = request.get_json(force=True)
    user_text = data.get("message", "").strip()

    if not user_text:
        return jsonify({"error": "No 'message' field provided."}), 400

    # ----- Sentiment -----
    vec = string2vec(glove_reps, user_text)
    if np.linalg.norm(vec) == 0.0:
        sentiment_reply = "I'm not sure how you're feeling based on that."
    else:
        label = model.predict(vec.reshape(1, -1))[0]
        if label == 0:
            sentiment_reply = "Hmm, it seems like you're feeling a bit down."
        else:
            sentiment_reply = "It sounds like you're in a positive mood!"

    # ----- Stylistic analysis -----
    dep = get_dependency_parse(user_text)
    nsubj, obj, iobj, nmod, amod = get_dep_categories(dep)
    feat = custom_feature(user_text)

    style_reply = (
        "Here's what I discovered about your writing style.\n"
        f"# Nominal Subjects: {nsubj}\n"
        f"# Direct Objects: {obj}\n"
        f"# Indirect Objects: {iobj}\n"
        f"# Nominal Modifiers: {nmod}\n"
        f"# Adjectival Modifiers: {amod}\n"
        f"Custom Feature: {feat}"
    )

    return jsonify(
        {
            "sentiment": sentiment_reply,
            "style": style_reply,
        }
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

