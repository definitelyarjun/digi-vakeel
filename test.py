from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key="d01df583-622b-4cc1-ac0e-205807f7f7f7")

response = client.text.translate(
input="भारत एक महान देश है। इसकी संस्कृति बहुत पुरानी और समृद्ध है।",
source_language_code="hi-IN",
target_language_code="en-IN",
model="sarvam-translate:v1",

)
print(response.translated_text)
