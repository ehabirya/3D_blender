# app/calibration.py
"""
Calibration module
- Height mandatory; other measurements optional (used for camera/pose sanity + downstream fit)
- Analyzes uploaded photos using shared vision.py (Pose + FaceMesh, CPU-only)
- Quality gates photos and produces localized retake tips (EN/ES/DE/TR/FR)
- Reports which role (front/side/back) passed/failed and why
- Chooses best front/side/back candidate for downstream Blender step
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

# Shared vision utils (make sure app/vision.py exists)
from vision import (
    b64_to_img, analyze_one, quality_ok, choose_roles, DEFAULT_THRESHOLDS
)

# =========================
# Numeric helpers
# =========================

def _to_float(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None

def _cm_to_m(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    # meters if <=3; else cm→m
    return v if v <= 3.0 else v / 100.0


# =========================
# Language + localized tips
# =========================

SUPPORTED_LANGS = {"en", "es", "de", "tr", "fr"}

def _normalize_lang(code: str | None) -> str:
    if not code:
        return "en"
    code = code.strip().lower()
    if code in SUPPORTED_LANGS:
        return code
    two = code[:2]
    return two if two in SUPPORTED_LANGS else "en"

def resolve_lang(data: dict) -> str:
    """
    Resolve language from:
      - data.lang / data.language
      - data.locale (e.g., 'de-DE')
      - data.accept_language (e.g., 'fr-FR,fr;q=0.9,en;q=0.8')
    """
    cand = (
        data.get("lang")
        or data.get("language")
        or data.get("locale")
        or data.get("accept_language")
        or ""
    )
    if isinstance(cand, str) and "," in cand:
        cand = cand.split(",")[0]
    return _normalize_lang(cand)

TIPS_L10N = {
    "en": {
        "blur": "Hold the phone steady and tap to focus on the body. Use good lighting.",
        "shoulders": "Step back a little and make sure both shoulders are fully in frame.",
        "tilt": "Hold the phone level (no tilt). Keep the frame upright.",
        "front_face": "For the front photo, face the camera directly with shoulders level.",
        "side_turn": "For the side photo, turn 90° to the camera and keep posture straight.",
        "too_close": "You’re too close—step back about 0.5–1 m for a full-body shot.",
        "too_far": "You’re too far—move closer about 0.5–1 m so your body fills the frame.",
        "generic": "Stand straight in good light; keep the whole body visible and the phone level."
    },
    "es": {
        "blur": "Sujeta el móvil firme y toca para enfocar el cuerpo. Usa buena iluminación.",
        "shoulders": "Aléjate un poco y asegúrate de que ambos hombros estén dentro del encuadre.",
        "tilt": "Mantén el móvil nivelado (sin inclinación). Mantén el encuadre recto.",
        "front_face": "Para la foto frontal, mira a la cámara con los hombros rectos.",
        "side_turn": "Para la foto lateral, gírate 90° hacia la cámara y mantén la postura recta.",
        "too_close": "Estás demasiado cerca: aléjate unos 0,5–1 m para una toma de cuerpo entero.",
        "too_far": "Estás demasiado lejos: acércate unos 0,5–1 m para que el cuerpo llene el encuadre.",
        "generic": "Ponte recto con buena luz; cuerpo completo visible y el móvil nivelado."
    },
    "de": {
        "blur": "Halte das Handy ruhig und tippe zum Fokussieren auf den Körper. Gute Beleuchtung hilft.",
        "shoulders": "Geh einen Schritt zurück und achte darauf, dass beide Schultern im Bild sind.",
        "tilt": "Halte das Handy waagrecht (nicht kippen). Rahmen gerade halten.",
        "front_face": "Für die Frontaufnahme: direkt zur Kamera schauen, Schultern gerade.",
        "side_turn": "Für die Seitenaufnahme: 90° zur Kamera drehen und aufrechte Haltung.",
        "too_close": "Du bist zu nah – geh etwa 0,5–1 m zurück für eine Ganzkörperaufnahme.",
        "too_far": "Du bist zu weit weg – geh etwa 0,5–1 m näher heran, damit der Körper das Bild füllt.",
        "generic": "Stell dich aufrecht in gutes Licht; ganzer Körper sichtbar, Handy gerade halten."
    },
    "tr": {
        "blur": "Telefonu sabit tutun ve gövdeye odaklamak için dokunun. İyi aydınlatma kullanın.",
        "shoulders": "Biraz geri çekilin ve her iki omuzun da kadraja girdiğinden emin olun.",
        "tilt": "Telefonu düz tutun (eğmeyin). Kadrajı dik tutun.",
        "front_face": "Ön fotoğraf için kameraya doğrudan bakın, omuzlar düz olsun.",
        "side_turn": "Yan fotoğraf için kameraya 90° dönün ve dik durun.",
        "too_close": "Çok yakınsınız — tam boy çekim için 0,5–1 m geri adım atın.",
        "too_far": "Çok uzaksınız — beden kadrajı doldursun diye 0,5–1 m yaklaşın.",
        "generic": "İyi ışıkta dik durun; tüm beden görünür ve telefon düz olsun."
    },
    "fr": {
        "blur": "Tenez le téléphone bien stable et touchez pour faire la mise au point sur le corps. Bonne lumière conseillée.",
        "shoulders": "Reculez un peu et assurez-vous que les deux épaules sont dans le cadre.",
        "tilt": "Tenez le téléphone droit (sans inclinaison). Gardez le cadre vertical.",
        "front_face": "Pour la photo de face, regardez la caméra avec les épaules alignées.",
        "side_turn": "Pour la photo de profil, tournez-vous à 90° et gardez une posture droite.",
        "too_close": "Vous êtes trop près — reculez d’environ 0,5 à 1 m pour un plein pied.",
        "too_far": "Vous êtes trop loin — rapprochez-vous d’environ 0,5 à 1 m pour remplir le cadre.",
        "generic": "Tenez-vous droit, bonne lumière; corps entier visible et téléphone bien droit."
    },
}

def reason_to_tip_localized(reason: str, analysis: dict, lang: str) -> str:
    t = TIPS_L10N.get(lang, TIPS_L10N["en"])
    r = reason.lower()
    dist = (analysis or {}).get("camera", {}).get("distance_m")

    if "low focus" in r:
        return t["blur"]
    if "shoulders not fully visible" in r or "subject too far" in r:
        return t["shoulders"]
    if "camera tilt" in r:
        return t["tilt"]
    if "not facing camera enough for front view" in r:
        return t["front_face"]
    if "not turned enough for side view" in r:
        return t["side_turn"]

    if isinstance(dist, (int, float)):
        if dist < 1.2:
            return t["too_close"]
        if dist > 3.0:
            return t["too_far"]

    return t["generic"]

def tips_for_rejected_localized(entry: dict, lang: str) -> list[str]:
    tips: list[str] = []
    for reason in entry.get("reasons", []):
        tip = reason_to_tip_localized(reason, entry.get("analysis", {}), lang)
        if tip not in tips:
            tips.append(tip)
    if not tips:
        tips.append(TIPS_L10N.get(lang, TIPS_L10N["en"])["generic"])
    return tips


# =========================
# Role reporting helpers
# =========================

def _collect_role_labels(photos_dict) -> list[str | None]:
    """
    Build a role label list aligned to b64_list order:
    - if explicit photos.front/side/back: record the label
    - if photos.unordered: label is None
    """
    if not isinstance(photos_dict, dict):
        return []
    labels = []
    if isinstance(photos_dict.get("unordered"), list) and photos_dict["unordered"]:
        labels = [None for _ in photos_dict["unordered"]]
    else:
        for k in ("front", "side", "back"):
            if isinstance(photos_dict.get(k), str):
                labels.append(k)
    return labels

def _build_role_report(roles=("front","side","back"),
                       chosen_by_role=None,
                       accepted=None,
                       rejected=None,
                       index_to_b64=None,
                       provided_labels=None,
                       lang="en"):
    """
    Per-role status:
      status: 'ok' (good photo), 'retry' (candidate failed thresholds), 'missing' (no candidate)
      chosen_index: index in original input order (or None)
      provided_indices: indices user provided for that role (if labeled input)
      failed_indices: subset of provided_indices that were rejected
      reasons, tips: from the chosen/rejected candidate (if any)
    """
    chosen_by_role = chosen_by_role or {}
    accepted = accepted or []
    rejected = rejected or []
    provided_labels = provided_labels or []
    index_to_b64 = index_to_b64 or {}

    b64_to_index = {b64: idx for idx, b64 in index_to_b64.items() if isinstance(b64, str)}

    provided_idx_by_role = {r: [] for r in roles}
    for i, label in enumerate(provided_labels):
        if label in roles:
            provided_idx_by_role[label].append(i)

    acc_idx = {e["index"] for e in accepted}
    rej_idx = {e["index"] for e in rejected}

    report = {}
    for role in roles:
        chosen_b64 = chosen_by_role.get(role)
        chosen_idx = b64_to_index.get(chosen_b64, None)

        status = "missing"
        reasons = []
        tips = []
        failed_indices = []

        if chosen_idx is not None:
            if chosen_idx in acc_idx:
                status = "ok"
            elif chosen_idx in rej_idx:
                status = "retry"
                r = next((e for e in rejected if e["index"] == chosen_idx), None)
                if r:
                    reasons = r.get("reasons", [])[:]
                    tips = r.get("tips", [])[:]
            else:
                status = "retry"

        for idx in provided_idx_by_role.get(role, []):
            if idx in rej_idx:
                failed_indices.append(idx)

        report[role] = {
            "status": status,
            "chosen_index": chosen_idx,
            "provided_indices": provided_idx_by_role.get(role, []),
            "failed_indices": failed_indices,
            "reasons": reasons,
            "tips": tips
        }
    return report


# =========================
# Main API
# =========================

def calibrate_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shared calibration:
      - height (required) normalized to meters
      - optional measurements normalized
      - analyze all photos via vision.py
      - apply quality thresholds → accepted/rejected with reasons + localized tips
      - choose best front/side/back even if none accepted (so pipeline can still proceed)
      - build role_report to show which role failed and why
    """
    # Resolve language once
    lang = resolve_lang(data)

    # 1) Height (required)
    height = _to_float(data.get("height"))
    if height is None:
        return {"ok": False, "error": "height is required (cm or m)", "lang": lang}
    height_m = _cm_to_m(height)
    if height_m < 0.3 or height_m > 2.6:
        return {"ok": False, "error": f"unrealistic height: {height_m:.2f} m", "lang": lang}

    # 2) Optional measurements (meters)
    def cm(key): return _cm_to_m(_to_float(data.get(key)))
    chest    = cm("chest")
    waist    = cm("waist")
    hips     = cm("hips")
    shoulder = cm("shoulder")
    inseam   = cm("inseam")
    arm      = cm("arm")
    foot_len = cm("foot_length") or cm("footLength")
    foot_wid = cm("foot_width")  or cm("footWidth")
    foot_cat = (data.get("foot_width_category") or data.get("footWidthCategory") or "").strip() or None

    out: Dict[str, Any] = {
        "ok": True,
        "lang": lang,
        "height_m": height_m,
        "chest": chest, "waist": waist, "hips": hips,
        "shoulder": shoulder, "inseam": inseam, "arm": arm,
        "foot_length_m": foot_len, "foot_width_m": foot_wid,
        "foot_width_category": foot_cat,
        "thresholds": DEFAULT_THRESHOLDS
    }

    # 3) Gather photos (unordered list preferred; else explicit roles)
    photos = (data.get("photos") or {})
    unordered = photos.get("unordered") if isinstance(photos, dict) else None

    b64_list: List[str] = []
    if isinstance(unordered, list) and unordered:
        b64_list.extend([b for b in unordered if isinstance(b, str)])
    else:
        for k in ("front", "side", "back"):
            v = photos.get(k) if isinstance(photos, dict) else None
            if isinstance(v, str):
                b64_list.append(v)

    if not b64_list:
        out.update({
            "accepted": [],
            "rejected": [],
            "chosen_by_role": {"front": None, "side": None, "back": None},
            "retake_tips": [],
            "role_report": {
                "front": {"status":"missing","chosen_index":None,"provided_indices":[],"failed_indices":[],"reasons":[],"tips":[]},
                "side":  {"status":"missing","chosen_index":None,"provided_indices":[],"failed_indices":[],"reasons":[],"tips":[]},
                "back":  {"status":"missing","chosen_index":None,"provided_indices":[],"failed_indices":[],"reasons":[],"tips":[]},
            }
        })
        return out

    # Keep maps for role reporting
    index_to_b64 = {i: b for i, b in enumerate(b64_list)}
    provided_labels = _collect_role_labels(photos)

    # 4) Analyze all photos (shared logic)
    analyses: List[Dict[str, Any]] = []
    for b64 in b64_list:
        img = b64_to_img(b64)
        if img is None:
            analyses.append({"decode_error": True, "role": "unknown"})
            continue
        analyses.append(analyze_one(img, height_m))

    # 5) Quality gate with shared thresholds
    accepted, rejected = [], []
    for idx, (b64, a) in enumerate(zip(b64_list, analyses)):
        role_hint = None  # if you want: set to provided_labels[idx] when explicit roles are given
        ok, reasons = quality_ok(a, role_hint)
        entry = {"index": idx, "role_pred": a.get("role"), "reasons": reasons, "analysis": a}
        if ok:
            accepted.append(entry)
        else:
            entry["tips"] = tips_for_rejected_localized(entry, lang)
            rejected.append(entry)

    # 6) Choose best front/side/back
    if accepted:
        chosen = choose_roles([a["analysis"] for a in accepted],
                              [b64_list[e["index"]] for e in accepted])["by_role"]
    else:
        chosen = choose_roles(analyses, b64_list)["by_role"]

    # 7) Dedup global retake tips
    retake_tips: List[str] = []
    for r in rejected:
        for tip in r.get("tips", []):
            if tip not in retake_tips:
                retake_tips.append(tip)

    # 8) Role report (which role failed & why)
    role_report = _build_role_report(
        roles=("front","side","back"),
        chosen_by_role=chosen,
        accepted=accepted,
        rejected=rejected,
        index_to_b64=index_to_b64,
        provided_labels=provided_labels,
        lang=lang
    )

    out.update({
        "accepted": accepted,
        "rejected": rejected,
        "chosen_by_role": chosen,
        "retake_tips": retake_tips,
        "role_report": role_report
    })
    return out
