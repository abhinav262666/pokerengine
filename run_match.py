from poker_agent import PokerEngine, Phase
from agent_llm import AgentLLM


def run_multi_player_match():
    # 4 players:
    player_ids = ["gpt4o_A", "gpt4omini", "gpt4o_B", "o3mini"]

    models = {
        "gpt4o_A": "gpt-4o",
        "gpt4omini": "gpt-4o-mini",
        "gpt4o_B": "gpt-4o",        # second instance of GPT-4o
        "o3mini": "o3-mini"         # OpenAI o3-mini
    }

    engine = PokerEngine(player_ids, starting_stacks=[500, 500, 500, 500])
    engine.start_hand(deck_seed="demo-seed-4player")
    print("=== INITIAL STATE ===")
    print(engine.serialize())

    agents = {
        pid: AgentLLM(models[pid], pid)
        for pid in player_ids
    }

    print("=== Starting 4-player match ===")

    # Main game loop
    while engine.phase != Phase.COMPLETED:

        # Whose turn is it?
        current_pid = engine.players[engine.current_actor_index].player_id
        agent = agents[current_pid]

        # Build state
        player_obj = next(p for p in engine.players if p.player_id == current_pid)
        if player_obj.is_all_in or player_obj.is_folded:
            engine._advance_turn()
            continue
        
        state = engine.canonical_state_for(current_pid)
        legal = engine.legal_actions(current_pid)

        # Ask the LLM for an action
        action, amount = agent.decide(state, legal)
        print(f"{current_pid} → {action} {amount}")

        result = engine.apply_action(current_pid, action, amount)
        print(result)

        # If hand ended after this action
        if engine.phase == Phase.COMPLETED:
            break

        if "error" in result:
            print("Engine error:", result)
            break

    print("\n=== FINAL RESULT ===")
    print(engine.serialize())


if __name__ == "__main__":
    run_multi_player_match()
