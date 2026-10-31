from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_token: str = ""
    github_webhook_secret: str = ""
    github_api_url: str = "https://api.github.com"
    publish_reviews: bool = False
    review_categories: str = "security,bug,performance,dependency,impact"
    max_inline_comments: int = 15
    repo_context_max_files: int = 300
    max_context_tokens: int = 2000
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "qwen/qwen3.8-27b"
    database_url: str = "postgresql://review:review@db:5432/review"


settings = Settings()
