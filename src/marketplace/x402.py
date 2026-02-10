"""x402 Payment Layer — HTTP 402 payment gate and escrow for agent marketplace.

Implements the x402 V2 protocol:
- 402 Payment Required responses with structured accepts array
- Payment verification
- Escrow for hire-complete-release lifecycle

x402 V2 response format:
  HTTP 402 Payment Required
  Header: X-Payment: required
  Body: {
    "error": "Payment Required",
    "accepts": [{
      "scheme": "exact",
      "network": "eip155:8453",
      "maxAmountRequired": "0",
      "resource": "<url>",
      "description": "...",
      "mimeType": "application/json",
      "payTo": "0x...",
      "requiredDeadlineSeconds": 300,
      "outputSchema": null,
      "extra": {
        "name": "...",
        "facilitatorUrl": "https://facilitator.payai.network"
      }
    }]
  }
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PaymentConfig:
    """Configuration for x402 payment requirements."""

    network: str = "eip155:8453"  # Base mainnet
    price: float = 0.0  # USDC amount
    pay_to: str = ""  # Receiver wallet address
    asset: str = "USDC"
    facilitator_url: str = "https://facilitator.payai.network"
    deadline_seconds: int = 300

    def to_accepts_entry(self, resource: str = "", description: str = "") -> dict[str, Any]:
        """Generate a single entry for the x402 'accepts' array."""
        # Convert price to smallest unit string (USDC has 6 decimals)
        amount_micro = str(int(self.price * 1_000_000))
        return {
            "scheme": "exact",
            "network": self.network,
            "maxAmountRequired": amount_micro,
            "resource": resource,
            "description": description,
            "mimeType": "application/json",
            "payTo": self.pay_to,
            "requiredDeadlineSeconds": self.deadline_seconds,
            "outputSchema": None,
            "extra": {
                "name": self.asset,
                "facilitatorUrl": self.facilitator_url,
            },
        }


@dataclass
class PaymentProof:
    """Proof of payment from a payer."""

    payment_id: str = field(default_factory=lambda: f"pay_{uuid.uuid4().hex[:12]}")
    payer: str = ""
    payee: str = ""
    amount: float = 0.0
    network: str = "eip155:8453"
    tx_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    verified: bool = False


class X402PaymentGate:
    """Generates 402 responses and verifies payment proofs.

    Used by marketplace endpoints to gate access behind x402 payment.
    """

    def __init__(self, config: PaymentConfig | None = None) -> None:
        self._config = config or PaymentConfig()
        self._payments: list[PaymentProof] = []
        self._verified_resources: dict[str, PaymentProof] = {}

    @property
    def config(self) -> PaymentConfig:
        return self._config

    def create_402_response(
        self,
        resource: str = "",
        description: str = "Payment required to access this agent service",
        price_override: float | None = None,
    ) -> dict[str, Any]:
        """Generate a 402 Payment Required response body.

        Matches the x402 V2 spec format with accepts array.
        """
        config = self._config
        if price_override is not None:
            # Temporary config with overridden price
            config = PaymentConfig(
                network=self._config.network,
                price=price_override,
                pay_to=self._config.pay_to,
                asset=self._config.asset,
                facilitator_url=self._config.facilitator_url,
                deadline_seconds=self._config.deadline_seconds,
            )

        return {
            "error": "Payment Required",
            "accepts": [config.to_accepts_entry(resource, description)],
        }

    def verify_payment(self, proof: PaymentProof) -> bool:
        """Verify a payment proof.

        In production, this would verify on-chain. For testing,
        we check amount and payee match.
        """
        expected_payee = self._config.pay_to
        if expected_payee and proof.payee != expected_payee:
            return False
        if proof.amount < self._config.price:
            return False

        proof.verified = True
        self._payments.append(proof)
        return True

    def record_verified_payment(self, resource: str, proof: PaymentProof) -> None:
        """Record a verified payment for a resource."""
        self._verified_resources[resource] = proof

    def is_paid(self, resource: str) -> bool:
        """Check if a resource has been paid for."""
        return resource in self._verified_resources

    def payment_history(self, payer: str | None = None) -> list[PaymentProof]:
        """Get payment history, optionally filtered by payer."""
        if payer is None:
            return list(self._payments)
        return [p for p in self._payments if p.payer == payer]

    def total_collected(self) -> float:
        """Total USDC collected from verified payments."""
        return sum(p.amount for p in self._payments if p.verified)


@dataclass
class EscrowEntry:
    """An escrow hold for an agent hiring."""

    escrow_id: str = field(default_factory=lambda: f"escrow_{uuid.uuid4().hex[:12]}")
    payer: str = ""
    payee: str = ""
    amount: float = 0.0
    task_id: str = ""
    status: str = "held"  # "held", "released", "refunded"
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None


class AgentEscrow:
    """Escrow system for agent marketplace payments.

    Flow: hold_payment() -> work happens -> release_on_completion() or refund_on_failure()
    """

    def __init__(self) -> None:
        self._entries: dict[str, EscrowEntry] = {}

    def hold_payment(
        self,
        payer: str,
        payee: str,
        amount: float,
        task_id: str,
    ) -> EscrowEntry:
        """Create an escrow hold for a task payment."""
        entry = EscrowEntry(
            payer=payer,
            payee=payee,
            amount=amount,
            task_id=task_id,
            status="held",
        )
        self._entries[entry.escrow_id] = entry
        return entry

    def release_on_completion(self, escrow_id: str) -> EscrowEntry | None:
        """Release escrowed funds to the payee on task completion."""
        entry = self._entries.get(escrow_id)
        if entry is None or entry.status != "held":
            return None
        entry.status = "released"
        entry.resolved_at = time.time()
        return entry

    def refund_on_failure(self, escrow_id: str) -> EscrowEntry | None:
        """Refund escrowed funds to the payer on task failure."""
        entry = self._entries.get(escrow_id)
        if entry is None or entry.status != "held":
            return None
        entry.status = "refunded"
        entry.resolved_at = time.time()
        return entry

    def get_entry(self, escrow_id: str) -> EscrowEntry | None:
        """Get an escrow entry by ID."""
        return self._entries.get(escrow_id)

    def get_entries_for_task(self, task_id: str) -> list[EscrowEntry]:
        """Get all escrow entries for a task."""
        return [e for e in self._entries.values() if e.task_id == task_id]

    def list_held(self) -> list[EscrowEntry]:
        """List all currently held escrow entries."""
        return [e for e in self._entries.values() if e.status == "held"]

    def list_all(self) -> list[EscrowEntry]:
        """List all escrow entries."""
        return list(self._entries.values())

    def total_held(self) -> float:
        """Total USDC currently held in escrow."""
        return sum(e.amount for e in self._entries.values() if e.status == "held")

    def total_released(self) -> float:
        """Total USDC released from escrow."""
        return sum(e.amount for e in self._entries.values() if e.status == "released")

    def clear(self) -> None:
        """Clear all escrow entries."""
        self._entries.clear()


__all__ = [
    "PaymentConfig",
    "PaymentProof",
    "X402PaymentGate",
    "EscrowEntry",
    "AgentEscrow",
]
