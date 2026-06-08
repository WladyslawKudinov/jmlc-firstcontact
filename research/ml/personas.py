import numpy as np

LATENT_TRAITS = ["readiness", "urgency", "budget_fit", "program_fit", "engagement"]
# Trait->outcome signal is calibrated so a full-trait oracle reaches AUC ~0.78 — the
# predictability of real lead-scoring systems. (The original weights were ~2x weaker,
# making lead quality a near coin-flip / oracle ~0.66, which contradicted the product
# premise that leads are rankable.) Weights and BIAS scale together so the base rate
# stays 50%. Genuine irreducible error remains via the Gaussian noise + Bernoulli draw.
TRAIT_WEIGHTS = {"readiness": 4.4, "urgency": 2.8, "budget_fit": 2.4, "program_fit": 2.0, "engagement": 1.2}
BIAS = 6.4
LABEL_NOISE_SD = 0.4

def sample_latent(rng):
    return {t: float(rng.beta(2.0, 2.0)) for t in LATENT_TRAITS}

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit(theta):
    return sum(TRAIT_WEIGHTS[t] * theta[t] for t in LATENT_TRAITS) - BIAS

def label_probability(theta):
    return float(_sigmoid(_logit(theta)))

def sample_label(theta, rng):
    noisy = _logit(theta) + float(rng.normal(0.0, LABEL_NOISE_SD))
    return int(rng.random() < _sigmoid(noisy))

_BANDS = {
    "readiness": ("ты просто собираешь информацию, решения пока нет",
                  "ты присматриваешься, но окончательно не решил(а)",
                  "ты уже почти решил(а) учиться, спрашиваешь про старт"),
    "urgency": ("со сроками не торопишься, «когда-нибудь потом»",
                "готов(а) начать в ближайший месяц",
                "хочешь начать как можно скорее, буквально на этой неделе"),
    "budget_fit": ("бюджет ограничен, переживаешь о цене, просишь рассрочку/скидку",
                   "цена важна, спросишь про стоимость и варианты",
                   "цена для тебя не проблема, про деньги не переживаешь"),
    "program_fit": ("твой запрос лишь частично совпадает с программой школы",
                    "твой запрос в целом подходит школе",
                    "твой запрос точно совпадает с тем, что школа предлагает"),
    "engagement": ("отвечаешь односложно, неохотно, коротко",
                   "отвечаешь по делу, нейтрально",
                   "отвечаешь развёрнуто, охотно, задаёшь встречные вопросы"),
}

def _band(v):
    return 0 if v < 0.34 else (2 if v > 0.66 else 1)

def behavior_brief(theta):
    return "\n".join("- " + _BANDS[t][_band(theta[t])] for t in LATENT_TRAITS)
