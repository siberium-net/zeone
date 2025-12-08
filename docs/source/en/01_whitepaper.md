# ZEONE Whitepaper

## Abstract

Modern internet infrastructure is built on centralized protocols (BGP, DNS, CA) that create single points of failure and enable mass surveillance and censorship by state actors. ZEONE is a decentralized operating system implementing the Compute-over-Data paradigm: computation moves to data, not vice versa. The system unifies P2P transport, distributed AI inference (Cortex), VPN routing, and cryptoeconomics into a single sovereign network layer.

---

## 1. Philosophy and Threat Landscape

### 1.1 BGP Vulnerabilities

Border Gateway Protocol (RFC 4271) is the inter-domain routing protocol that holds the entire internet together. Its fundamental problems:

**BGP Hijacking:**
Any autonomous system (AS) can announce foreign IP prefixes. Route origin verification (RPKI) is deployed on less than 40% of networks. An attacker can:
- Redirect traffic through their AS for interception
- Create a blackhole for DoS
- Execute MITM on unencrypted connections

**AS-Path Manipulation:**
BGP trusts the AS_PATH attribute without cryptographic verification. An attacker can:
- Shorten the path to attract traffic
- Add fake AS to bypass filters
- Use BGP communities to manipulate routing policies

**Notable Incidents:**
- 2018: Amazon Route 53 hijack for stealing $150K in cryptocurrency
- 2022: Russia Telecom redirecting Twitter/Facebook traffic

### 1.2 DNS Problems

Domain Name System (RFC 1035) is a hierarchical system with root servers under ICANN control.

**Root Server Centralization:**
13 root servers (A-M) are controlled by organizations predominantly from the USA. Theoretically possible:
- TLD removal for political reasons
- Forced domain redirection
- Mass deanonymization through DNS query logs

**Cache Poisoning (RFC 5452):**
An attacker can inject false records into a resolver's DNS cache:
```
Query ID: 16-bit (65536 variants)
Source Port: often predictable
→ Birthday attack: ~256 packets for 50% success
```

**DNS-over-HTTPS (DoH) ≠ solution:**
DoH encrypts the channel but centralizes resolvers (Cloudflare 1.1.1.1, Google 8.8.8.8). The DoH provider receives the complete user visit graph.

### 1.3 Certificate Authority Risks

PKI (Public Key Infrastructure) is built on trust in ~150 root CAs.

**CA Compromise:**
- 2011: DigiNotar — fake certificates issued for *.google.com
- 2015: CNNIC — subordinate CA issuing MITMing certificates
- 2016: WoSign — certificate backdating

**State-level MITM:**
States can compel national CAs to issue certificates for any domain. Certificate Transparency (CT) logs detect this post-factum but don't prevent it.

**Revocation Problem:**
OCSP/CRL checks are not always performed (soft-fail by default). Browsers cache status, creating a vulnerability window.

### 1.4 ZEONE Solution

ZEONE eliminates dependence on centralized protocols:

| Problem | Traditional Solution | ZEONE |
|---------|---------------------|-------|
| Routing | BGP (trust-based) | Kademlia DHT (cryptographic) |
| Naming | DNS (hierarchical) | Content-addressing (SHA-256) |
| Identity | CA (delegated trust) | Ed25519 keypairs (self-sovereign) |
| Encryption | TLS (CA-dependent) | NaCl Box (peer-to-peer) |

---

## 2. Trust Score Mathematics

### 2.1 Current Implementation

Trust Score \( T \in [0, 1] \) is a local peer reputation metric computed independently by each node.

**Basic Update Formula:**

$$
T_{new} = \text{clamp}(T_{old} + w_e \cdot m, 0, 1)
$$

where:
- \( w_e \) — weight of event \( e \) from the table
- \( m \) — magnitude (e.g., transaction size in MB)
- \( \text{clamp}(x, a, b) = \max(a, \min(b, x)) \)

**Event Weight Table:**

| Event | Weight \( w_e \) | Description |
|-------|------------------|-------------|
| `successful_transfer` | +0.01 | Successful data transfer |
| `failed_transfer` | -0.05 | Failed transfer |
| `valid_message` | +0.001 | Valid signed message |
| `invalid_message` | -0.02 | Invalid signature |
| `iou_created` | +0.005 | IOU created |
| `iou_redeemed` | +0.02 | IOU redeemed |
| `iou_defaulted` | -0.1 | IOU expired |
| `ping_responded` | +0.001 | Responded to PING |
| `ping_timeout` | -0.01 | Failed to respond to PING |
| `debt_repaid` | +0.02 | Debt repaid |
| `excessive_debt` | -0.03 | Debt limit exceeded |

### 2.2 Inactivity Decay

Reputation degrades exponentially with no interactions:

$$
T_{decayed} = T \cdot 0.99^{d}
$$

where \( d \) is the number of inactive days.

**Half-life:** \( t_{1/2} = \frac{\ln 2}{\ln(1/0.99)} \approx 69 \) days.

### 2.3 EMA Formula [IMPLEMENTED]

For smoother updates, Exponential Moving Average is used:

$$
T_{new} = \alpha \cdot R + (1 - \alpha) \cdot T_{old}
$$

where:
- \( R \in [0, 1] \) — result of last interaction (1 = success, 0 = failure)
- \( \alpha \in (0, 1) \) — smoothing coefficient (recommended 0.1)

**EMA Advantages:**
- Natural bound \( T \in [0, 1] \) without clamp
- Exponential forgetting of old events
- Parameter \( \alpha \) controls system "memory"

**[OK]** Implemented in [`economy/trust.py:206-253`](../../economy/trust.py) - `WeightedTrustScore.update_behavior_score()`

### 2.4 Initial Value and Cold Start

New peers receive:
- \( T_0 = 0.5 \) (neutral)
- Minimum interaction threshold: \( T_{min} = 0.1 \)

**Cold Start Protection:**
A peer with \( T < T_{min} \) can only:
1. Respond to PING (liveness proof)
2. Participate in DHT routing
3. Create IOUs with high collateral

---

## 3. Token Economics [PROPOSED]

### 3.1 ZEO Token

- **Standard:** Native SIBR token on Siberium blockchain
- **Network:** Siberium (ChainID 111111 mainnet, 111000 testnet)
- **Initial Supply:** 100,000,000 ZEO
- **Decimals:** 18
- **Gas Token:** SIBR (native, like ETH on Ethereum)

### 3.2 Emission Curve

A logarithmic curve with asymptotic limit is proposed:

$$
S(t) = S_{max} \cdot \left(1 - e^{-\lambda t}\right)
$$

where:
- \( S(t) \) — total supply at time \( t \)
- \( S_{max} = 1,000,000,000 \) ZEO — maximum supply
- \( \lambda = 0.1 \) year⁻¹ — emission rate

**Emission in year \( n \):**

$$
\Delta S_n = S_{max} \cdot e^{-\lambda n} \cdot (1 - e^{-\lambda})
$$

| Year | Emission (M ZEO) | Cumulative (M ZEO) |
|------|------------------|---------------------|
| 1    | 95.16            | 95.16               |
| 2    | 86.07            | 181.23              |
| 3    | 77.86            | 259.09              |
| 5    | 63.76            | 393.47              |
| 10   | 40.66            | 632.12              |

### 3.3 Burn Mechanism

A portion of service fees (VPN, CDN, AI) is burned on each transaction:

$$
\text{burn} = \text{fee} \cdot \beta
$$

where \( \beta = 0.2 \) (20% of fee is burned).

**Deflationary Pressure:**
At high network utilization, burning can exceed emission, creating deflationary dynamics.

### 3.4 Exit Node Staking

VPN Exit and CDN Provider nodes must stake ZEO:

$$
\text{stake}_{min} = \text{bandwidth}_{Mbps} \cdot k_{stake}
$$

where \( k_{stake} = 100 \) ZEO/Mbps.

**Slashing Conditions:**
- Invalid data (hash mismatch): -10% stake
- Unavailability > 1 hour: -1% stake
- Content censorship: -50% stake + ban

---

## 4. Threat Model

### 4.1 Sybil Attack

**Description:** Attacker creates multiple fake identities to capture network control.

**ZEONE Protection:**

1. **Identity Cost:**
   - Ed25519 keypair generation: ~0.1ms on modern CPU
   - Insufficient for economic barrier

2. **Trust Score cold start:**
   - New nodes start with \( T_0 = 0.5 \)
   - Time required to accumulate reputation
   - Sybil nodes cannot instantly gain influence

3. **Stake requirement [IMPLEMENTED]:**
   - For DHT routing participation: minimum stake 10 ZEO
   - For service provision: stake proportional to bandwidth
   
**[OK]** Implemented via `WeightedTrustScore` in [`economy/trust.py`](../../economy/trust.py).

Formula: `T_effective = T_behavior * log10(1 + Stake / BaseStake)`

**Effectiveness:**
With stake requirement = 10 ZEO and price 1 ZEO = $0.10:
- Cost of 1000 Sybil nodes: $1,000
- Cost of attacking network with 10,000 nodes (>50%): $50,000+

### 4.2 Eclipse Attack

**Description:** Attacker isolates victim by filling their routing table with malicious nodes.

**ZEONE Protection (Kademlia):**

1. **K-bucket diversity:**
   - \( K = 20 \) nodes per bucket
   - 160 buckets → up to 3200 unique nodes in table
   - Attacker needs to control nodes in all relevant buckets

2. **Random bucket refresh:**
   - Every hour: lookup random ID in each bucket
   - Updates table with nodes from different parts of ID space

3. **LRU with verification:**
   - On bucket overflow: PING oldest node
   - New node replaces only if old one doesn't respond
   - Protects against legitimate node eviction

**Attack Mathematics:**
For eclipse with probability \( p \):
- Need to control \( \geq K \cdot \log_2(N) \) nodes
- With \( N = 10000, K = 20 \): ~266 nodes for partial eclipse

### 4.3 Free-riding (Leeching)

**Description:** Node consumes network resources without contributing.

**ZEONE Protection:**

1. **Ledger blocking:**
   ```python
   debt_limit = 100 * 1024 * 1024  # 100 MB
   
   if balance[peer] > debt_limit:
       block_sending_to(peer)
   ```

2. **Tit-for-Tat in Amplifier:**
   - Node A gives chunk to node B
   - B owes A equivalent volume
   - On limit exceeded: B blocked until repayment

3. **Trust Score penalty:**
   - `excessive_debt`: -0.03 to Trust Score
   - With \( T < 0.1 \): node cannot request data

### 4.4 Replay Attack

**Description:** Attacker intercepts and re-sends a valid message.

**Protection:**

1. **Timestamp validation:**
   ```python
   MAX_MESSAGE_AGE = 60  # seconds
   if time.time() - message.timestamp > MAX_MESSAGE_AGE:
       reject(message)
   ```

2. **Nonce uniqueness:**
   - Each message contains 16-byte random nonce
   - Node stores seen nonces in bloom filter (1 hour TTL)
   - Repeated nonce → rejection

3. **Session keys:**
   - After handshake: unique session key via ECDH
   - Messages bound to session

---

## 5. Governance [PROPOSED]

### 5.1 DAO Structure

Network parameter governance through decentralized voting:

**Governed Parameters:**
- Fees: `fee_vpn`, `fee_cdn`, `fee_ai`
- Limits: `debt_limit`, `max_stake`
- Policies: `min_trust_score`, `slashing_rate`
- Cortex model whitelist

### 5.2 Voting Mechanism

**Vote Weight:**

$$
V = \sqrt{S} \cdot (1 + T)
$$

where:
- \( S \) — stake in ZEO
- \( T \) — Trust Score

**Square root** of stake prevents plutocracy.

**Process:**
1. Proposal submission: requires 1000 ZEO deposit
2. Discussion period: 7 days
3. Voting period: 7 days
4. Execution delay: 2 days (for exit)

---

## Conclusion

ZEONE proposes a fundamentally new internet architecture that eliminates dependence on centralized trust points. Cryptographic identity, economic incentives, and distributed storage create a censorship-resistant infrastructure for computation, communication, and data storage.

---

## References

1. RFC 4271 — A Border Gateway Protocol 4 (BGP-4)
2. RFC 1035 — Domain Names - Implementation and Specification
3. RFC 5452 — Measures for Making DNS More Resilient against Forged Answers
4. Maymounkov, P., Mazières, D. — Kademlia: A Peer-to-peer Information System Based on the XOR Metric
5. Bernstein, D.J. — Curve25519: new Diffie-Hellman speed records
6. Nakamoto, S. — Bitcoin: A Peer-to-Peer Electronic Cash System
