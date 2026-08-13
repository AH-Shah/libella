import torch
import inspect

from libella.utils import scatter_softmax
from libella.model import LibellaGNN

print("\n" + "="*40)
print("🔍 RUNNING LIBELLA GNN SAFETY CHECKS")
print("="*40)

# ---------------------------------------------------------
# TEST 1: The 60.0 Hardware Clamp (utils.py)
# ---------------------------------------------------------
# We feed it 4 values. 
# Node 0 gets [10.0, 20.0] (Normal values, should be exact math)
# Node 1 gets [70.0, 100.0] (Over 60.0! Should clamp to [60, 60] and output 0.5 / 0.5)
src = torch.tensor([10.0, 20.0, 70.0, 100.0])
idx = torch.tensor([0, 0, 1, 1])
out = scatter_softmax(src, idx, 2)

print("\n▶ TEST 1: Fast MPS Softmax (utils.py)")
print(f"  Inputs:  [10.0, 20.0] | [70.0, 100.0]")
print(f"  Outputs: {[round(x, 5) for x in out.tolist()]}")

if abs(out[2].item() - 0.5) < 0.01:
    print("  ✅ SUCCESS: Hardware clamp is active! (70 and 100 were correctly flattened)")
else:
    print("  ❌ FAILED: Softmax did exact math. The new utils.py is not loaded.")


# ---------------------------------------------------------
# TEST 2: The 0.05 Safety Net (model.py)
# ---------------------------------------------------------
# We dynamically read the installed source code of your GNN
# to ensure the safety floor is present in the GATv2 block.
source_code = inspect.getsource(LibellaGNN.encode)

print("\n▶ TEST 2: GATv2 Safety Net (model.py)")
if "min=0.05" in source_code:
    print("  ✅ SUCCESS: 'min=0.05' is active in LibellaGNN.encode!")
elif "min=0.1" in source_code:
    print("  ✅ SUCCESS: 'min=0.1' is active in LibellaGNN.encode!")
else:
    print("  ❌ FAILED: Could not find the 0.05 safety floor in your model code.")

print("\n" + "="*40 + "\n")