import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import enum, uuid

# ---- Hand evaluator (Treys) ----
from treys import Evaluator, Card

evaluator = Evaluator()

class Phase(enum.Enum):
    PRE_FLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    COMPLETED = "completed"

@dataclass
class PlayerState:
    player_id: str
    stack: int
    hole_cards: List[str] = field(default_factory=list)
    committed: int = 0
    is_folded: bool = False
    is_all_in: bool = False
    seat: int = 0

@dataclass
class Action:
    player_id: str
    action: str
    amount: int = 0

class PokerEngine:
    def __init__(self, player_ids: List[str], starting_stacks=None, small_blind=5, big_blind=10):
        assert len(player_ids) >= 2

        if starting_stacks is None:
            starting_stacks = [1000] * len(player_ids)

        self.players: List[PlayerState] = [
            PlayerState(player_id=pid, stack=starting_stacks[i], seat=i)
            for i, pid in enumerate(player_ids)
        ]

        self.phase = None
        self.button_index = 0
        self.current_actor_index = 0
        self.deck = []
        self.community_cards = []
        self.hand_id = str(uuid.uuid4())
        self.pot_history = []
        self.actions = []
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.total_pot = 0
        self.deck_seed = None

        self._phase_order = [
            Phase.PRE_FLOP,
            Phase.FLOP,
            Phase.TURN,
            Phase.RIVER,
            Phase.SHOWDOWN,
            Phase.COMPLETED
        ]

    # ---- Deck ----
    @staticmethod
    def standard_deck():
        ranks = "23456789TJQKA"
        suits = "cdhs"
        return [r + s for r in ranks for s in suits]

    @staticmethod
    def shuffle_with_seed(seed: str):
        deck = PokerEngine.standard_deck()
        rnd = random.Random(seed)
        rnd.shuffle(deck)
        return deck

    # ---- Start Hand ----
    def start_hand(self, deck_seed=None):
        self.deck_seed = deck_seed or str(uuid.uuid4())
        self.deck = self.shuffle_with_seed(self.deck_seed)
        self.community_cards = []
        self.total_pot = 0
        self.actions = []
        self.pot_history = []

        for p in self.players:
            p.hole_cards = []
            p.committed = 0
            p.is_folded = False
            p.is_all_in = False

        sb = self.button_index
        bb = (self.button_index + 1) % len(self.players)

        def post_blind(idx, amount, label):
            p = self.players[idx]
            actual = min(p.stack, amount)
            p.stack -= actual
            p.committed += actual
            self.total_pot += actual
            self.actions.append(Action(p.player_id, f"post_{label}", actual))
            if p.stack == 0:
                p.is_all_in = True

        post_blind(sb, self.small_blind, "sb")
        post_blind(bb, self.big_blind, "bb")

        deal_start = (self.button_index + 1) % len(self.players)
        for _ in range(2):
            for i in range(len(self.players)):
                seat = (deal_start + i) % len(self.players)
                card = self.deck.pop(0)
                self.players[seat].hole_cards.append(card)

        self.phase = Phase.PRE_FLOP
        self.current_actor_index = (bb + 1) % len(self.players)

        # BUT if that player is all-in or folded, skip to next available
        n = len(self.players)
        for i in range(n):
            idx = (self.current_actor_index + i) % n
            p = self.players[idx]
            if not p.is_folded and not p.is_all_in:
                self.current_actor_index = idx
                break

        return {
            "hand_id": self.hand_id,
            "deck_seed": self.deck_seed
        }

    # ---- Helpers ----
    def find_player_index(self, player_id):
        for i, p in enumerate(self.players):
            if p.player_id == player_id:
                return i
        raise KeyError(player_id)

    def canonical_state_for(self, player_id):
        return {
            "hand_id": self.hand_id,
            "phase": self.phase.value if self.phase else None,
            "community_cards": list(self.community_cards),
            "your_hole_cards": list(self.players[self.find_player_index(player_id)].hole_cards),
            "players": [
                {
                    "player_id": p.player_id,
                    "stack": p.stack,
                    "committed": p.committed,
                    "is_folded": p.is_folded,
                    "is_all_in": p.is_all_in
                }
                for p in self.players
            ],
            "total_pot": self.total_pot,
            "to_act_player_id":
                self.players[self.current_actor_index].player_id
                if self.phase and self.phase != Phase.COMPLETED
                else None,
            "deck_seed": self.deck_seed,
        }

    # ---- Legal Actions ----
    def legal_actions(self, player_id: str) -> List[str]:
        idx = self.find_player_index(player_id)
        p = self.players[idx]

        if p.is_folded or p.is_all_in or self.phase == Phase.COMPLETED:
            return []

        highest = max(pl.committed for pl in self.players)
        to_call = highest - p.committed

        # If there is money to call, NO CHECK is allowed.
        if to_call > 0:
            actions = ["fold"]
            if p.stack >= to_call:
                actions.append("call")
            if p.stack > to_call:
                actions.append("raise")
            if p.stack > 0:
                actions.append("allin")
            return actions

        # No one has bet more → check is allowed
        actions = ["check"]
        if p.stack > 0:
            actions.append("bet")
            actions.append("allin")
        return actions


    # ---- Apply Action ----
    def apply_action(self, player_id, action, amount=0):
        if self.phase in (None, Phase.COMPLETED):
            return {"error": "No active hand"}

        idx = self.find_player_index(player_id)
        p = self.players[idx]

        if p.is_folded or p.is_all_in:
            return {"error": "Player cannot act"}

        legal = self.legal_actions(player_id)
        if action not in legal:
            return {"error": f"Illegal action: {action}. Allowed: {legal}"}

        highest = max(pl.committed for pl in self.players)
        to_call = highest - p.committed

        if action == "fold":
            p.is_folded = True
            self.actions.append(Action(player_id, action, 0))

        elif action == "call":
            pay = min(p.stack, to_call)
            p.stack -= pay
            p.committed += pay
            self.total_pot += pay
            if p.stack == 0:
                p.is_all_in = True
            self.actions.append(Action(player_id, action, pay))

        elif action == "check":
            self.actions.append(Action(player_id, action, 0))

        elif action == "bet":
            if amount <= 0 or amount > p.stack:
                return {"error": "Invalid bet amount"}
            p.stack -= amount
            p.committed += amount
            self.total_pot += amount
            if p.stack == 0:
                p.is_all_in = True
            self.actions.append(Action(player_id, "bet", amount))

        elif action == "raise":
            if amount <= 0 or amount > p.stack:
                return {"error": "Invalid raise amount"}
            pay = min(p.stack, to_call + amount)
            p.stack -= pay
            p.committed += pay
            self.total_pot += pay
            if p.stack == 0:
                p.is_all_in = True
            self.actions.append(Action(player_id, "raise", pay))

        elif action == "allin":
            pay = p.stack
            p.stack = 0
            p.committed += pay
            self.total_pot += pay
            p.is_all_in = True
            self.actions.append(Action(player_id, "allin", pay))

        # turn moves
        self._advance_turn()

        # end of round?
        if self._is_betting_round_complete():
            self._advance_phase()

        return {
            "ok": True,
            "phase": self.phase.value,
            "total_pot": self.total_pot
        }

    # ---- Turn rotation ----
    def _advance_turn(self):
        # Determine if any player can act
        active_players = [
            p for p in self.players
            if not p.is_folded and not p.is_all_in
        ]

        # If no one can act: force betting round complete
        if len(active_players) == 0:
            return

        n = len(self.players)
        for i in range(1, n + 1):
            idx = (self.current_actor_index + i) % n
            p = self.players[idx]
            if not p.is_folded and not p.is_all_in:
                self.current_actor_index = idx
                return



    # ---- Betting round complete ----
    def _is_betting_round_complete(self) -> bool:
        # Count players who can act
        active = [p for p in self.players if not p.is_folded and not p.is_all_in]

        # Everyone folded or all-in → round is complete
        if len(active) == 0:
            return True

        highest = max(p.committed for p in self.players)

        # If any active player has not matched the highest, round not done
        for p in active:
            if p.committed != highest:
                return False

        # All active players matched → round done
        return True


    # ---- Deal streets ----
    def _advance_phase(self):
        cur = self._phase_order.index(self.phase)
        next_phase = self._phase_order[cur + 1]
        self.phase = next_phase

        # --- Normal street dealing ---
        if self.phase == Phase.FLOP:
            _ = self.deck.pop(0)
            self.community_cards += [self.deck.pop(0) for _ in range(3)]

        elif self.phase == Phase.TURN:
            _ = self.deck.pop(0)
            self.community_cards.append(self.deck.pop(0))

        elif self.phase == Phase.RIVER:
            _ = self.deck.pop(0)
            self.community_cards.append(self.deck.pop(0))

        elif self.phase == Phase.SHOWDOWN:
            self.evaluate_and_distribute()
            self.phase = Phase.COMPLETED
            return

        # --- Auto-run remaining streets if everyone is all-in/folded ---
        active = [
            p for p in self.players
            if not p.is_folded and not p.is_all_in
        ]

        # If no one left to act, fast-forward the rest of the hand
        if len(active) == 0:
            while self.phase not in (Phase.SHOWDOWN, Phase.COMPLETED):
                cur = self._phase_order.index(self.phase)
                next_phase = self._phase_order[cur + 1]
                self.phase = next_phase

                if self.phase == Phase.TURN:
                    _ = self.deck.pop(0)
                    self.community_cards.append(self.deck.pop(0))

                elif self.phase == Phase.RIVER:
                    _ = self.deck.pop(0)
                    self.community_cards.append(self.deck.pop(0))

                elif self.phase == Phase.SHOWDOWN:
                    self.evaluate_and_distribute()
                    self.phase = Phase.COMPLETED
                    break


    # ---- EVALUATION (REAL WINNER SELECTION) ----
    def evaluate_and_distribute(self):
        active = [p for p in self.players if not p.is_folded]

        if len(active) == 1:
            # only one player left → wins whole pot
            winner = active[0]
            winner.stack += self.total_pot
            self.pot_history.append({
                "winner": winner.player_id,
                "amount": self.total_pot
            })
            self.total_pot = 0
            return

        board = [Card.new(c) for c in self.community_cards]

        scores = []
        for p in active:
            hand = [Card.new(c) for c in p.hole_cards]
            score = evaluator.evaluate(board, hand)
            scores.append((score, p))

        scores.sort(key=lambda x: x[0])
        best_score, winner = scores[0]

        # find ties (equal score)
        tied = [p for s, p in scores if s == best_score]

        share = self.total_pot // len(tied)
        remainder = self.total_pot % len(tied)

        for p in tied:
            p.stack += share

        # give remainder to first in seat order
        tied[0].stack += remainder

        self.pot_history.append({
            "winners": [p.player_id for p in tied],
            "split": share,
            "remainder": remainder
        })

        self.total_pot = 0

    # ---- Serialize ----
    def serialize(self):
        return {
            "hand_id": self.hand_id,
            "phase": self.phase.value if self.phase else None,
            "players": [
                {
                    "player_id": p.player_id,
                    "stack": p.stack,
                    "committed": p.committed,
                    "is_folded": p.is_folded,
                    "is_all_in": p.is_all_in,
                    "hole_cards": p.hole_cards
                }
                for p in self.players
            ],
            "community_cards": self.community_cards,
            "total_pot": self.total_pot,
            "deck_seed": self.deck_seed
        }
