from app.core.config import settings


print("App name:", settings.app_name)
print("App version:", settings.app_version)
print("Log level:", settings.log_level)
print("AI provider:", settings.ai_provider)
print("HF base URL:", settings.hf_base_url)
print("HF model:", settings.hf_model)
print("Ollama URL:", settings.ollama_url)
print("Ollama model:", settings.ollama_model)
print("AI request timeout (seconds):", settings.ai_request_timeout_seconds)
print("Default credits:", settings.default_credits)
print("Max input length:", settings.max_input_length)
print("Min input length:", settings.min_input_length)
print("Characters per credit:", settings.characters_per_credit)
print("JWT algorithm:", settings.jwt_algorithm)
print("JWT expiry (minutes):", settings.jwt_expiry_minutes)

print("HF token loaded:", bool(settings.hf_token))
print("JWT secret key loaded:", bool(settings.jwt_secret_key))
print("Auth client ID loaded:", bool(settings.auth_client_id))
print("Auth client secret loaded:", bool(settings.auth_client_secret))
