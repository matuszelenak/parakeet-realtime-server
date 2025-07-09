from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STABLE_ITERATION_COUNT: int = 3
    STABLE_WORDS_PER_ITERATION: int = 3
    MIN_TRANSCRIBED_DURATION: float = 0.5
    MAX_TRANSCRIBED_DURATION: float = 5.0

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
