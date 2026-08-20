import traceback
import sys
from main import compile_local_dataset

try:
    compile_local_dataset()
except Exception as e:
    print("EXCEPTION OCCURRED:", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
print("COMPLETED SUCCESSFULLY", file=sys.stderr)
