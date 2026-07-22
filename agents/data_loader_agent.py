"""Data loading agent for Excel and CSV files."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from core.base_agent import BaseAgent
from core.context import AgentContext
from utils.dataframe_helpers import detect_datetime_columns
from utils.exceptions import DataLoadError
from utils.logging_config import get_logger

logger = get_logger("data_loader")


class DataLoaderAgent(BaseAgent):
    """Load Excel / CSV sources into a validated pandas DataFrame."""

    name = "data_loader"
    description = "Excel ve CSV dosyalarını otomatik okur"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Load a dataset from path or uploaded file-like object.

        Args:
            context: Shared context.
            **kwargs: Must include ``source``. Optional: sheet_name, encoding,
                delimiter.

        Returns:
            AgentContext: Context with dataframe populated.

        Raises:
            DataLoadError: If the source cannot be read.
        """
        source = kwargs.get("source")
        if source is None:
            raise DataLoadError("source parametresi zorunludur.")

        sheet_name = kwargs.get("sheet_name")
        encoding = kwargs.get("encoding")
        delimiter = kwargs.get("delimiter")

        try:
            file_name, df, meta = self._load(source, sheet_name, encoding, delimiter)
            datetime_cols = detect_datetime_columns(df)
            for col in datetime_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Could not parse datetime column %s: %s", col, exc)

            context.dataframe = df
            context.raw_dataframe = df.copy()
            context.file_name = file_name
            context.metadata["load"] = meta
            context.metadata["datetime_columns"] = datetime_cols
            context.add_message(
                f"Dosya yüklendi: {file_name} ({df.shape[0]} satır, {df.shape[1]} sütun)"
            )
            self.logger.info(
                "Loaded %s with shape %s | meta=%s",
                file_name,
                df.shape,
                meta,
            )
            return context
        except DataLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(f"Dosya okunamadı: {exc}") from exc

    def _load(
        self,
        source: Union[str, Path, Any],
        sheet_name: Optional[Any],
        encoding: Optional[str],
        delimiter: Optional[str],
    ) -> tuple[str, pd.DataFrame, Dict[str, Any]]:
        """Internal loader supporting paths and Streamlit UploadedFile objects.

        Args:
            source: Path or file-like.
            sheet_name: Excel sheet name/index.
            encoding: Optional CSV encoding.
            delimiter: Optional CSV delimiter.

        Returns:
            tuple: file_name, dataframe, metadata.

        Raises:
            DataLoadError: On unsupported or corrupt files.
        """
        if hasattr(source, "name") and hasattr(source, "getvalue"):
            file_name = str(source.name)
            payload = source.getvalue()
            return self._load_bytes(file_name, payload, sheet_name, encoding, delimiter)

        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"Dosya bulunamadı: {path}")
        payload = path.read_bytes()
        return self._load_bytes(path.name, payload, sheet_name, encoding, delimiter)

    def _load_bytes(
        self,
        file_name: str,
        payload: bytes,
        sheet_name: Optional[Any],
        encoding: Optional[str],
        delimiter: Optional[str],
    ) -> tuple[str, pd.DataFrame, Dict[str, Any]]:
        """Load dataframe from raw bytes.

        Args:
            file_name: Original file name.
            payload: File bytes.
            sheet_name: Excel sheet selector.
            encoding: CSV encoding override.
            delimiter: CSV delimiter override.

        Returns:
            tuple: file_name, dataframe, metadata.

        Raises:
            DataLoadError: If parsing fails.
        """
        suffix = Path(file_name).suffix.lower()
        meta: Dict[str, Any] = {"file_name": file_name, "suffix": suffix}

        if suffix in {".xlsx", ".xls", ".xlsm"}:
            return self._load_excel(file_name, payload, sheet_name, meta)
        if suffix in {".csv", ".txt", ".tsv"}:
            return self._load_csv(file_name, payload, encoding, delimiter, meta, suffix)
        raise DataLoadError(
            f"Desteklenmeyen dosya türü: {suffix}. CSV veya Excel yükleyin."
        )

    def _load_excel(
        self,
        file_name: str,
        payload: bytes,
        sheet_name: Optional[Any],
        meta: Dict[str, Any],
    ) -> tuple[str, pd.DataFrame, Dict[str, Any]]:
        """Load an Excel workbook.

        Args:
            file_name: File name.
            payload: Bytes.
            sheet_name: Optional sheet.
            meta: Metadata dict to enrich.

        Returns:
            tuple: file_name, dataframe, metadata.

        Raises:
            DataLoadError: If Excel is corrupt or empty.
        """
        try:
            buffer = io.BytesIO(payload)
            excel = pd.ExcelFile(buffer)
            sheets: List[str] = list(excel.sheet_names)
            meta["sheets"] = sheets
            selected = sheet_name if sheet_name is not None else sheets[0]
            if selected not in sheets and not isinstance(selected, int):
                raise DataLoadError(
                    f"Sayfa bulunamadı: {selected}. Mevcut sayfalar: {sheets}"
                )
            df = pd.read_excel(excel, sheet_name=selected)
            if df.empty:
                raise DataLoadError("Excel dosyası boş veya okunabilir veri içermiyor.")
            meta["selected_sheet"] = selected
            meta["rows"] = int(df.shape[0])
            meta["columns"] = int(df.shape[1])
            self.logger.info("Excel sheets detected: %s | selected=%s", sheets, selected)
            return file_name, df, meta
        except DataLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(
                f"Excel dosyası bozuk veya okunamadı: {exc}"
            ) from exc

    def _load_csv(
        self,
        file_name: str,
        payload: bytes,
        encoding: Optional[str],
        delimiter: Optional[str],
        meta: Dict[str, Any],
        suffix: str,
    ) -> tuple[str, pd.DataFrame, Dict[str, Any]]:
        """Load a CSV / TSV text file with encoding and delimiter detection.

        Args:
            file_name: File name.
            payload: Bytes.
            encoding: Optional encoding.
            delimiter: Optional delimiter.
            meta: Metadata dict.
            suffix: File suffix.

        Returns:
            tuple: file_name, dataframe, metadata.

        Raises:
            DataLoadError: If CSV cannot be decoded/parsed.
        """
        encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin-1", "cp1254", "iso-8859-9"]
        encodings = [e for e in encodings if e]
        default_delim = "\t" if suffix == ".tsv" else None
        delimiters = [delimiter] if delimiter else [default_delim, ",", ";", "|", "\t"]
        delimiters = [d for d in delimiters]

        last_error: Optional[Exception] = None
        for enc in encodings:
            for delim in delimiters:
                try:
                    text = payload.decode(enc)
                    read_kwargs: Dict[str, Any] = {}
                    if delim is not None:
                        read_kwargs["sep"] = delim
                    else:
                        read_kwargs["sep"] = None
                        read_kwargs["engine"] = "python"
                    df = pd.read_csv(io.StringIO(text), **read_kwargs)
                    if df.shape[1] <= 1 and delim in {",", ";", "|"}:
                        # Likely wrong delimiter; try next.
                        continue
                    if df.empty:
                        raise DataLoadError("CSV dosyası boş.")
                    meta.update(
                        {
                            "encoding": enc,
                            "delimiter": delim or "auto",
                            "rows": int(df.shape[0]),
                            "columns": int(df.shape[1]),
                        }
                    )
                    self.logger.info(
                        "CSV loaded with encoding=%s delimiter=%s shape=%s",
                        enc,
                        delim,
                        df.shape,
                    )
                    return file_name, df, meta
                except DataLoadError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue

        raise DataLoadError(
            f"CSV dosyası bozuk veya kodlama/ayırıcı algılanamadı: {last_error}"
        )
