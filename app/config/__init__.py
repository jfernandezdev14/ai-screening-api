from pydantic_settings import BaseSettings, SettingsConfigDict


class CommonSettings(BaseSettings):
    APP_NAME: str = "FARM Intro"
    DEBUG_MODE: bool = False


class ServerSettings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000

class FileSettings(BaseSettings):
    FILES_DIRECTORY: str = "./db/"


# class DatabaseSettings(BaseSettings):
#     DB_URL: str
#     DB_NAME: str

class Settings(CommonSettings, ServerSettings, FileSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


settings = Settings()
