from pydantic import BaseModel, HttpUrl


class ToolDownloadPlatformConfig(BaseModel):
    download_url: HttpUrl
    binary_name: str


class ToolDownloadConfig(BaseModel):
    Windows: ToolDownloadPlatformConfig
    Linux: ToolDownloadPlatformConfig
    Darwin: ToolDownloadPlatformConfig
