from guardrails.hub import DetectPII
from guardrails import Guard

guard = Guard().use(
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix")
)

text = "hi my name is aryan and my number is 00000000000"

try:
    result = guard.validate(text)
    print(result)
    print("text passed validation")
except Exception as e:
    print(f"error: {e}")
