from email import message_from_string
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================
# SYSTEM STRATEGY PROMPT
# ============================
SYSTEM_PROMPT = """
You are an autonomous poker agent playing Heads-Up No-Limit Texas Hold’em. 
Your job is to ALWAYS choose a strategically strong legal action AND a valid amount.

==============================
### ABSOLUTE ACTION RULES
==============================
1. NEVER fold preflop unless your hand is extremely weak *and* opponent has raised.
2. If legal_actions contains "raise":
   - Use this formula for raise amount:
     amount = max(to_call + round(stack * 0.3), to_call + 10)
3. If legal_actions contains "bet":
   - Bet = round(stack * 0.3)
4. If only "call" and "fold" exist:
   - Call unless your hand is complete trash AND opponent shows aggression.
5. If "check" exists → prefer "check" unless you have:
   - strong made hand → value bet
   - strong draw → semi-bluff
6. If "allin" is legal:
   Go all-in ONLY with: top pair good kicker, overpairs, 2-pair+, strong draws (≥12 outs).

==============================
### STRATEGIC FRAMEWORK
==============================
PRE-FLOP RANGES:
- Raise with: Any Ace, any King, any Queen, any Jack, any pair, any suited connector (54s+), any two broadways.
- Call with: 65o+, most suited hands, one-gappers, medium offsuit hands.
- Fold only trash: 32o, 42o, 83o, 93o.

POST-FLOP:
- C-bet often when you were aggressor.
- Value bet strong hands.
- Semi-bluff good draws.
- Check marginal hands.
- Fold weak air vs aggression.

==============================
### MATH CONSTRAINTS (IMPORTANT)
==============================
- ALL bet/raise amounts must be integers.
- Amount must NEVER exceed your stack.
- Amount must NEVER be negative.
- If amount isn't needed (fold/check/call), set amount = 0.

==============================
### OUTPUT FORMAT
==============================
Return ONLY:
{"action": "<action>", "amount": <int>}

No reasoning.
No commentary.
No words outside the JSON.
"""

# ============================
# AGENT CLASS
# ============================
class AgentLLM:
    def __init__(self, model_name: str, player_id: str):
        self.model = model_name
        self.player_id = player_id

    def decide(self, state: dict, legal_actions: list[str]):

        # USER PROMPT fed to LLM every decision
        prompt = f"""
GAME STATE:

Phase: {state['phase']}
Your ID: {self.player_id}

Your Hole Cards: {state['your_hole_cards']}
Community Cards: {state['community_cards']}

Pot: {state['total_pot']}
Players: {state['players']}
LEGAL ACTIONS: {legal_actions}

Your Stack: {[p for p in state['players'] if p['player_id']==self.player_id][0]['stack']}

To-Call amount: {max(pl['committed'] for pl in state['players']) - [p['committed'] for p in state['players'] if p['player_id']==self.player_id][0]}

Return JSON only.
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        if self.model.startswith("o3"):
            # o3 models DO NOT support temperature or max tokens
            response = client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        else:
            # GPT-4o and 4o-mini DO support temperature
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )


        # Extract model output
        content = response.choices[0].message.content.strip()

        # Parse JSON or fallback
        try:
            data = json.loads(content)
            return data["action"], data.get("amount", 0)
        except Exception:
            # fallback strategy – safe default
            return "check" if "check" in legal_actions else "fold", 0
