import sys
import os
from unittest.mock import patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'strategy_trainer'))

# We use patch to override the RESULTS_FILE path inside auto_loop just for this test
@patch('auto_loop.RESULTS_FILE', 'test_dummy_results.tsv')
def test_tsv_parsing():
    """Tests if the manager can read different whitespace formats without defaulting to -999."""
    import auto_loop
    
    # 1. Create a dummy TSV with mixed tabs and spaces (simulating the bug)
    with open('test_dummy_results.tsv', 'w') as f:
        f.write("commit    final_result    status    description\n") # spaces header
        f.write("abc1234\t-0.50\tkeep\tGood run\n")                 # tabs
        f.write("def5678    1.25    keep    Great run\n")            # spaces
        f.write("ghi9012\t-2.00\tdiscard\tBad run\n")               # tabs
        
    # 2. Run the parsing function
    best_score, history = auto_loop.get_history_and_best()
    
    # 3. Clean up the dummy file
    if os.path.exists('test_dummy_results.tsv'):
        os.remove('test_dummy_results.tsv')
        
    # 4. Assertions
    assert best_score == 1.25, f"Parser failed! Expected 1.25, got {best_score}"
    assert len(history) == 3, "Parser missed lines!"