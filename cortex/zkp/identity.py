"""
Prototype interfaces for future ZKP integration.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VerifiableCredential:
    issuer_did: str
    holder_did: str
    claim: Dict[str, Any]


class ZKVerifier:
    def verify_proof(self, proof: bytes, public_inputs: Dict[str, Any]) -> bool:
        """
        Placeholder verifier. TODO: Integrate groth16 verifier.
        """
        return True
