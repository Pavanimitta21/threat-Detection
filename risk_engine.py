def calculate_risk(prediction, confidence, indicators=0):

    if prediction in ["Threat", "Phishing", "threat"]:
        score = confidence
    else:
        score = 100 - confidence

    # Small increase when suspicious indicators are present
    score = score + (indicators * 5)

    # Keep score between 0 and 100
    score = min(round(score), 100)

    if score <= 25:
        level = "Low"
    elif score <= 50:
        level = "Medium"
    elif score <= 75:
        level = "High"
    else:
        level = "Critical"

    return score, level