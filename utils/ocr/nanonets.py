"""utils/ocr/nanonets.py — Moteur OCR Nanonets-OCR-s (VLM local).

Moteur OCR alternatif évalué pour les captures Reddit : Nanonets-OCR-s
(https://huggingface.co/nanonets/Nanonets-OCR-s), un modèle vision-langage
(Qwen2.5-VL 3B) qui lit une image de page et renvoie son texte.

Choix d'implémentation :
- modèle chargé une seule fois (lourd : ~7 Go) ;
- exécution locale sur MPS (Apple Silicon) si dispo, sinon CPU ;
- une fonction simple `ocr_image(pil_image) -> str` ; le rendu PDF->image et la
  boucle par page restent dans utils/data_loader.py (commune aux deux moteurs).

Aucun appel réseau au moment de l'import : tout est différé au premier OCR réel.
"""

import logging

# Identifiant du modèle Hugging Face (téléchargé/caché au premier usage).
MODEL_ID = "nanonets/Nanonets-OCR-s"

# Prompt recommandé par le modèle pour une extraction de texte fidèle.
OCR_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a small "
    "description of the image inside the <img></img> tag; otherwise, add the image caption "
    "inside <img></img>. Watermarks should be wrapped in brackets. "
    "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ "
    "and ☑ for check boxes."
)

# Plafond de tokens générés par page (une capture Reddit dépasse rarement ~1-2k tokens).
MAX_NEW_TOKENS = 4096

# État du modèle (chargé une seule fois).
_model = None
_processor = None
_device = None
_unavailable = False


def _select_device():
    """MPS (GPU Apple) si disponible, sinon CPU. Surchargeable par NANONETS_DEVICE."""
    import os
    import torch

    forced = os.getenv("NANONETS_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    """Charge le modèle Nanonets-OCR-s et son processor (une seule fois)."""
    global _model, _processor, _device, _unavailable
    if _model is not None or _unavailable:
        return _model, _processor
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _device = _select_device()
        # float16 sur MPS/GPU (plus rapide, moins de mémoire) ; float32 sur CPU (stable).
        dtype = torch.float16 if _device in ("mps", "cuda") else torch.float32
        logging.info(f"Chargement du modèle Nanonets-OCR-s ({MODEL_ID}) sur {_device} ({dtype})... (~7 Go au premier lancement)")
        _model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
        _model = _model.to(_device)
        _model.eval()
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        logging.info("Modèle Nanonets-OCR-s chargé.")
    except Exception as e:  # dépendance absente, téléchargement impossible, OOM...
        logging.warning(f"Nanonets-OCR-s indisponible ({e}). Moteur Nanonets ignoré.")
        _unavailable = True
        _model = None
        _processor = None
    return _model, _processor


def is_available():
    """Tente de charger le modèle et indique s'il est utilisable."""
    model, _ = _load()
    return model is not None


def ocr_image(pil_image):
    """Renvoie le texte reconnu par Nanonets-OCR-s sur une image de page (PIL).

    Retourne une chaîne (éventuellement vide). Ne lève pas : en cas d'erreur, log + "".
    """
    model, processor = _load()
    if model is None:
        return ""
    try:
        import torch

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_image], padding=True, return_tensors="pt")
        inputs = inputs.to(_device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        # On ne décode QUE les tokens générés (on retire le prompt d'entrée).
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded[0].strip() if decoded else ""
    except Exception as e:
        logging.error(f"Erreur OCR Nanonets sur une page : {e}")
        return ""
