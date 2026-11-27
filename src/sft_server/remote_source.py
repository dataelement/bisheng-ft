"""Download a training dataset straight from the customer's FTP / FTPS / SFTP / S3 storage.

The platform only records the manifest and every referenced file's size; the bytes travel from the
source to this host. Connection secrets arrive with a single job request, stay in memory and are never
written to disk or to logs.
"""
from __future__ import annotations

import base64
import ftplib
import hashlib
import ipaddress
import posixpath
import shutil
import socket
import ssl
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

CHUNK = 1024 * 1024
SECRET_FIELDS = ("password", "access_key", "secret_key", "session_token")


class RemoteSourceError(RuntimeError):
    """Messages name the failing step and path, never the connector's raw text (it can echo secrets)."""


def public_view(source: dict) -> dict:
    return {key: value for key, value in source.items() if key not in SECRET_FIELDS}


def _endpoint(source: dict):
    try:
        parsed = urlsplit(str(source.get("endpoint", "")))
    except ValueError:
        raise RemoteSourceError("invalid data source endpoint") from None
    schemes = {"ftp": {"ftp", "ftps"}, "sftp": {"sftp"}, "s3": {"http", "https"}}.get(source.get("type"))
    if (
        not schemes
        or parsed.scheme not in schemes
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise RemoteSourceError("invalid data source endpoint")
    try:
        port = parsed.port or {"ftp": 21, "sftp": 22}.get(source["type"]) or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except (ValueError, OSError):
        raise RemoteSourceError("data source host cannot be resolved") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            raise RemoteSourceError("loopback, metadata and reserved addresses are not allowed")
    return parsed, port, infos[0][4][0]


def remote_path(source: dict, relative: str) -> str:
    """Join a manifest-relative media path onto the configured root without allowing traversal."""
    member = PurePosixPath(relative)
    if not relative or member.is_absolute() or ".." in member.parts or "\\" in relative:
        raise RemoteSourceError(f"unsafe media reference: {relative}")
    root = str(source.get("media_root") or "").rstrip("/")
    return posixpath.join(root, relative) if root else relative


class _Ftp:
    def __init__(self, source, parsed, port, address):
        tls = bool(source.get("tls")) or parsed.scheme == "ftps"
        client = ftplib.FTP_TLS(context=ssl.create_default_context()) if tls else ftplib.FTP()
        # Pin the validated address while keeping the hostname for TLS verification.
        client.host = parsed.hostname
        client.timeout = 30
        client.sock = socket.create_connection((address, port), timeout=30)
        client.af = client.sock.family
        client.file = client.sock.makefile("r", encoding=client.encoding)
        client.welcome = client.getresp()
        client.login(source.get("username") or "anonymous", source.get("password") or "")
        if tls:
            client.prot_p()
        client.set_pasv(True)
        client.voidcmd("TYPE I")
        self.client = client

    def size(self, path):
        return self.client.size(path)

    def download(self, path, handle):
        self.client.retrbinary("RETR " + path, handle.write, blocksize=CHUNK)

    def close(self):
        try:
            self.client.quit()
        except ftplib.all_errors:
            self.client.close()


class _Sftp:
    def __init__(self, source, parsed, port, address):
        import paramiko

        if not source.get("username"):
            raise RemoteSourceError("SFTP username is required")
        self.transport = paramiko.Transport(socket.create_connection((address, port), timeout=30))
        try:
            self.transport.banner_timeout = 30
            self.transport.auth_timeout = 30
            self.transport.start_client(timeout=30)
            expected = (source.get("host_fingerprint") or "").strip().rstrip("=")
            if expected:
                key = self.transport.get_remote_server_key().asbytes()
                actual = "SHA256:" + base64.b64encode(hashlib.sha256(key).digest()).decode().rstrip("=")
                if actual != expected:
                    raise RemoteSourceError("SFTP host fingerprint does not match")
            self.transport.auth_password(source["username"], source.get("password") or "")
            self.client = paramiko.SFTPClient.from_transport(self.transport)
            self.client.get_channel().settimeout(120)
        except BaseException:
            self.transport.close()
            raise

    def size(self, path):
        return self.client.stat(path).st_size

    def download(self, path, handle):
        with self.client.open(path, "rb") as remote:
            remote.prefetch()
            while chunk := remote.read(CHUNK):
                handle.write(chunk)

    def close(self):
        self.transport.close()


class _S3:
    def __init__(self, source, parsed, port, address):
        from minio import Minio

        if not source.get("bucket"):
            raise RemoteSourceError("S3 bucket is required")
        self.bucket = source["bucket"]
        self.client = Minio(
            parsed.netloc,
            access_key=source.get("access_key") or None,
            secret_key=source.get("secret_key") or None,
            session_token=source.get("session_token") or None,
            secure=parsed.scheme == "https",
            region=source.get("region") or "us-east-1",
        )
        self.client.disable_virtual_style_endpoint()

    def size(self, path):
        return self.client.stat_object(self.bucket, path).size

    def download(self, path, handle):
        response = self.client.get_object(self.bucket, path)
        try:
            for chunk in response.stream(CHUNK):
                handle.write(chunk)
        finally:
            response.close()
            response.release_conn()

    def close(self):
        pass


@contextmanager
def open_source(source: dict):
    parsed, port, address = _endpoint(source)
    kind = {"ftp": _Ftp, "sftp": _Sftp, "s3": _S3}[source["type"]]
    try:
        session = kind(source, parsed, port, address)
    except RemoteSourceError:
        raise
    except Exception:
        raise RemoteSourceError(f"cannot connect to the {source['type'].upper()} data source; check address, account and network access") from None
    try:
        yield session
    finally:
        session.close()


def check(source: dict, probe: str) -> int:
    """Connect and read one file's size, so a wrong password or blocked network fails before training."""
    with open_source(source) as session:
        try:
            return session.size(probe)
        except Exception:
            raise RemoteSourceError(f"data source is reachable but cannot read {probe}") from None


def fetch(source: dict, files: list[dict], destination: Path, byte_limit: int, cancelled=None) -> int:
    """Download each planned file under destination and verify it still has the size recorded at import."""
    total = sum(int(item["size"]) for item in files)
    if total > byte_limit:
        raise RemoteSourceError("dataset exceeds the configured size limit of this training host")
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < total + 1024 ** 3:
        raise RemoteSourceError("not enough free disk space on the training host for this dataset")
    with open_source(source) as session:
        for item in files:
            if cancelled and cancelled():
                raise RemoteSourceError("job cancelled while downloading the dataset")
            target = destination / item["local"]
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open("wb") as handle:
                    session.download(item["remote"], handle)
            except Exception:
                raise RemoteSourceError(f"cannot download {item['remote']} from the data source") from None
            if target.stat().st_size != int(item["size"]):
                raise RemoteSourceError(f"{item['remote']} changed after import; import the dataset again")
    return total
