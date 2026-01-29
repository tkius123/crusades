"""Blockchain commitment reading for Templar Crusades.

URL-Based Architecture with Timelock Encryption:
- Miners host their train.py code at any URL (Gist, raw GitHub, etc.)
- Miners commit the code URL to blockchain using set_reveal_commitment()
- Commitments are timelock encrypted via drand
- After reveal_blocks, validators can read decrypted URL
- Validator fetches code from URL and evaluates

Commitment format:
  {
    "code_url": "https://example.com/train.py"
  }
"""

import ipaddress
import json
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import bittensor as bt

logger = logging.getLogger(__name__)

# Maximum size for commitment data to prevent DoS via huge JSON payloads
MAX_COMMITMENT_SIZE = 4096  # 4KB should be plenty for a URL

# Private/internal IP ranges to block (SSRF protection)
BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("10.0.0.0/8"),  # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),  # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),  # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / Cloud metadata
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 private
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

# Allowlisted domains for code hosting (for future use - domain allowlisting)
# Currently not enforced, but available for stricter security if needed
ALLOWED_DOMAINS = [
    "gist.githubusercontent.com",
    "raw.githubusercontent.com",
    "gist.github.com",
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "pastebin.com",
]


def is_ip_blocked(ip_str: str) -> bool:
    """Check if an IP address is in a blocked private/internal range.

    Args:
        ip_str: IP address as string

    Returns:
        True if the IP is blocked, False if allowed
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in network for network in BLOCKED_NETWORKS)
    except ValueError:
        # Invalid IP address format
        return True


def resolve_and_validate_host(hostname: str) -> tuple[bool, str]:
    """Resolve hostname to IP and validate it's not a blocked address.

    Args:
        hostname: The hostname to resolve

    Returns:
        Tuple of (is_valid, error_message_or_ip)
    """
    try:
        # Resolve hostname to IP address
        ip = socket.gethostbyname(hostname)

        if is_ip_blocked(ip):
            return False, f"Host {hostname} resolves to blocked IP range ({ip})"

        return True, ip
    except socket.gaierror as e:
        return False, f"Failed to resolve hostname {hostname}: {e}"
    except Exception as e:
        return False, f"Error validating host {hostname}: {e}"


@dataclass
class CodeUrlInfo:
    """Code URL information from miner commitment."""

    url: str

    def is_valid(self) -> bool:
        """Check if code URL is valid (basic format check only).

        Note: Full SSRF validation is done by validate_url_security() which
        resolves the hostname and checks for blocked IP ranges.
        """
        return bool(self.url) and (
            self.url.startswith("http://") or self.url.startswith("https://")
        )

    def validate_url_security(self) -> tuple[bool, str]:
        """Validate URL for SSRF protection by resolving hostname and checking IP ranges.

        This performs DNS resolution and checks that the resolved IP is not in
        a private/internal network range that could be used for SSRF attacks.

        Returns:
            Tuple of (is_safe, error_message). If is_safe is True, error_message
            contains the resolved IP. If is_safe is False, error_message contains
            the reason for rejection.
        """
        if not self.url:
            return False, "URL is empty"

        if not (self.url.startswith("http://") or self.url.startswith("https://")):
            return False, "URL must start with http:// or https://"

        try:
            parsed = urlparse(self.url)
            hostname = parsed.hostname

            if not hostname:
                return False, "URL has no hostname"

            # Check if hostname is already an IP address
            try:
                ip = ipaddress.ip_address(hostname)
                if is_ip_blocked(str(ip)):
                    return False, f"Direct IP {ip} is in blocked range"
                return True, str(ip)
            except ValueError:
                # Not an IP address, need to resolve hostname
                pass

            # Resolve and validate the hostname
            return resolve_and_validate_host(hostname)

        except Exception as e:
            return False, f"URL validation error: {e}"


@dataclass
class MinerCommitment:
    """A miner's commitment from the blockchain.

    Contains code URL for validator to fetch miner's train.py.
    """

    uid: int
    hotkey: str
    code_url_info: CodeUrlInfo | None
    reveal_block: int
    is_revealed: bool
    raw_data: str

    @classmethod
    def from_chain_data(
        cls,
        uid: int,
        hotkey: str,
        data: str,
        reveal_block: int,
        current_block: int,
    ) -> "MinerCommitment | None":
        """Parse commitment from blockchain data.

        Expects JSON format with code URL:
        {
            "code_url": "https://example.com/train.py"
        }

        Args:
            uid: Miner UID
            hotkey: Miner hotkey address
            data: Raw commitment data string (decrypted JSON)
            reveal_block: Block when commitment was revealed
            current_block: Current blockchain block

        Returns:
            MinerCommitment if valid, None if invalid format
        """
        if not data:
            return None

        data = data.strip()

        # Limit commitment size to prevent DoS via huge JSON payloads
        if len(data) > MAX_COMMITMENT_SIZE:
            logger.warning(
                f"Commitment from UID {uid} exceeds max size ({len(data)} > {MAX_COMMITMENT_SIZE}), skipping"
            )
            return None

        code_url_info = None

        # Parse JSON format
        if data.startswith("{"):
            try:
                parsed = json.loads(data)

                if "code_url" in parsed:
                    code_url_info = CodeUrlInfo(url=parsed["code_url"])
                    logger.debug(f"Code URL from UID {uid}: {code_url_info.url[:50]}...")

            except json.JSONDecodeError:
                logger.debug(f"Invalid JSON in commitment from UID {uid}")

        # If we couldn't parse code URL info, skip
        if code_url_info is None:
            logger.debug(f"Could not parse commitment from UID {uid}: {data[:50]}...")
            return None

        return cls(
            uid=uid,
            hotkey=hotkey,
            code_url_info=code_url_info,
            reveal_block=reveal_block,
            is_revealed=current_block >= reveal_block,
            raw_data=data,
        )

    def has_valid_code_url(self) -> bool:
        """Check if this commitment has a valid code URL."""
        return self.code_url_info is not None and self.code_url_info.is_valid()


class CommitmentReader:
    """Reads timelock-encrypted miner commitments from the Bittensor blockchain.

    Uses get_all_revealed_commitments() to read commitments that have been
    decrypted after their reveal block (set via set_reveal_commitment).
    """

    def __init__(
        self,
        subtensor: bt.subtensor | None = None,
        netuid: int = 1,
        network: str = "finney",
    ):
        """Initialize commitment reader.

        Args:
            subtensor: Bittensor subtensor instance
            netuid: Subnet ID
            network: Network name (local, test, finney)
        """
        self.netuid = netuid
        self.network = network
        self._subtensor = subtensor
        self._metagraph = None

    @property
    def subtensor(self) -> bt.subtensor:
        """Lazy-load subtensor connection."""
        if self._subtensor is None:
            self._subtensor = bt.subtensor(network=self.network)
        return self._subtensor

    @property
    def metagraph(self) -> bt.metagraph:
        """Lazy-load metagraph using subtensor (like templar)."""
        if self._metagraph is None:
            self._metagraph = self.subtensor.metagraph(netuid=self.netuid)
        return self._metagraph

    def sync(self) -> None:
        """Sync metagraph with blockchain."""
        logger.info(f"Syncing metagraph for subnet {self.netuid}...")
        # Re-fetch metagraph for fresh data (don't use .sync() - unreliable on localnet)
        self._metagraph = self.subtensor.metagraph(netuid=self.netuid)
        logger.info(f"Metagraph synced: {self._metagraph.n} neurons")

    def get_current_block(self) -> int:
        """Get current blockchain block number."""
        return self.subtensor.get_current_block()

    def _build_hotkey_to_uid_map(self) -> dict[str, int]:
        """Build mapping from hotkey to UID."""
        hotkey_to_uid: dict[str, int] = {}
        self.sync()
        for uid in range(self.metagraph.n):
            hotkey_to_uid[self.metagraph.hotkeys[uid]] = uid
        return hotkey_to_uid

    def _parse_revealed_result(self, result) -> tuple[int, str]:
        """Parse result from get_revealed_commitment APIs.

        Result format: ((reveal_block1, data1), (reveal_block2, data2), ...)
        - Tuple of tuples, one per commitment
        - Returns the LATEST commitment (highest block number)

        Returns:
            Tuple of (reveal_block, data) for the latest commitment
        """
        if not result:
            return 0, ""

        if isinstance(result, tuple):
            # Find the latest commitment (highest block number)
            latest_block = 0
            latest_data = ""

            for item in result:
                if isinstance(item, tuple) and len(item) >= 2:
                    block = item[0]
                    data = item[1] if item[1] else ""
                    if block > latest_block:
                        latest_block = block
                        latest_data = data

            if latest_block > 0:
                return latest_block, latest_data

        return 0, str(result) if result else ""

    def get_all_commitments(self) -> list[MinerCommitment]:
        """Get all revealed miner commitments from blockchain.

        Uses get_all_revealed_commitments() to read timelock-encrypted commits
        that have been decrypted after their reveal block.

        Returns:
            List of MinerCommitment objects with valid code URLs
        """
        current_block = self.get_current_block()
        commitments = []

        # Build hotkey to UID mapping (may fail on some localnet versions)
        hotkey_to_uid = self._build_hotkey_to_uid_map()

        # Get all revealed commitments (timelock decrypted)
        if hasattr(self.subtensor, "get_all_revealed_commitments"):
            try:
                logger.info("Reading revealed commitments from blockchain...")
                all_revealed = self.subtensor.get_all_revealed_commitments(netuid=self.netuid)
                logger.info(f"Found {len(all_revealed)} revealed commitments on chain")

                for hotkey, result in all_revealed.items():
                    uid = hotkey_to_uid.get(hotkey)
                    if uid is None:
                        logger.warning(
                            f"Hotkey {hotkey[:16]}... not found in metagraph, skipping commitment"
                        )
                        continue
                    reveal_block, data = self._parse_revealed_result(result)

                    commitment = MinerCommitment.from_chain_data(
                        uid=uid,
                        hotkey=hotkey,
                        data=data,
                        reveal_block=reveal_block,
                        current_block=current_block,
                    )
                    if commitment and commitment.has_valid_code_url():
                        commitment.is_revealed = True
                        commitments.append(commitment)
                        logger.debug(f"Valid commitment from {hotkey[:16]}... (UID {uid})")

                logger.info(f"Found {len(commitments)} valid commitments with code URLs")
                return commitments

            except Exception as e:
                logger.error(f"Failed to read revealed commitments: {e}")
        else:
            logger.error("Subtensor does not support get_all_revealed_commitments()")

        return commitments

    def get_commitment_for_hotkey(
        self,
        hotkey: str,
        current_block: int | None = None,
    ) -> MinerCommitment | None:
        """Get revealed commitment for a specific hotkey.

        Args:
            hotkey: Miner hotkey address
            current_block: Current block (fetched if None)

        Returns:
            MinerCommitment if found and valid, None otherwise
        """
        if current_block is None:
            current_block = self.get_current_block()

        # Get UID from metagraph
        uid = None
        try:
            hotkey_to_uid = self._build_hotkey_to_uid_map()
            uid = hotkey_to_uid.get(hotkey)
            if uid is None:
                logger.warning(f"Hotkey {hotkey[:16]}... not found in metagraph")
                return None
        except Exception as e:
            logger.warning(f"Failed to get UID for hotkey {hotkey[:16]}...: {e}")
            return None

        # Get revealed commitment
        if hasattr(self.subtensor, "get_revealed_commitment_by_hotkey"):
            try:
                result = self.subtensor.get_revealed_commitment_by_hotkey(
                    netuid=self.netuid,
                    hotkey_ss58_address=hotkey,
                )

                if result:
                    reveal_block, data = self._parse_revealed_result(result)

                    commitment = MinerCommitment.from_chain_data(
                        uid=uid,
                        hotkey=hotkey,
                        data=data,
                        reveal_block=reveal_block,
                        current_block=current_block,
                    )
                    if commitment:
                        commitment.is_revealed = True
                        return commitment

            except Exception as e:
                logger.debug(f"get_revealed_commitment_by_hotkey failed for {hotkey[:16]}...: {e}")

        return None

    def get_new_commitments_since(self, last_block: int) -> list[MinerCommitment]:
        """Get commitments revealed since a specific block.

        Args:
            last_block: Last processed block number

        Returns:
            List of newly revealed commitments
        """
        all_commitments = self.get_all_commitments()

        # Filter to only those revealed after last_block
        new_commitments = [c for c in all_commitments if c.reveal_block > last_block]

        logger.info(f"Found {len(new_commitments)} new commitments since block {last_block}")
        return new_commitments


def get_commitment_reader(
    network: str = "finney",
    netuid: int = 1,
) -> CommitmentReader:
    """Factory function to create a commitment reader.

    Args:
        network: Subtensor network
        netuid: Subnet ID

    Returns:
        Configured CommitmentReader
    """
    return CommitmentReader(netuid=netuid, network=network)
