import requests
import gradio as gr

# Azure Custom Vision -asetukset
PREDICTION_URL = "https://azureluokittelijainst-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/6f4236c0-5c75-406d-9d37-3350039ec246/classify/iterations/Iteration1/image"
HEADERS = {
    "Prediction-Key": "YOUR_PREDICTION_KEY_HERE",
    "Content-Type": "application/octet-stream"
}

def predict_image(image):
    try:
        with open(image, "rb") as f:
            response = requests.post(PREDICTION_URL, headers=HEADERS, data=f)
        response.raise_for_status()
        result = response.json()

        predictions = result.get("predictions", [])
        if not predictions:
            return "Ei tunnistettuja tageja."

        output_lines = []
        for pred in predictions:
            tag = pred.get("tagName", "")
            score = round(pred.get("probability", 0.0) * 100, 2)
            output_lines.append(f"{tag}: {score}%")

        best = max(predictions, key=lambda x: x.get("probability", 0.0))
        best_tag = best.get("tagName", "")
        best_score = round(best.get("probability", 0.0) * 100, 2)

        output_text = "\n".join(output_lines)
        return f"{output_text}\n\n➡️ Kuvassa todennäköisesti: {best_tag} ({best_score}%)"


    except Exception as e:
        return f"Virhe: {str(e)}"

# Gradio-käyttöliittymä
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Lataa kuva"),
    outputs="text",
    title="Sulo vai Kaapo?",
    description="Lataa kuva ja malli kertoo kumpi Sibbe siinä on. Näet myös todennäköisyydet kaikille tageille."
)

demo.launch()
