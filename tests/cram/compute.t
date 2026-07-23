  $ . "$TESTDIR/setup.sh"
  $ utrain compute list
  KIND  NAME                 CORES  POWER        COMPUTE  MEM_USED  MEM_TOTAL  MEM_PCT
  cpu   AMD Ryzen 9 7940HS   16     --           --       8.4       31.0       27.1%
  gpu   NVIDIA RTX 2000 Ada  --     9.5/140.0 W  8%       0.1       8.0        1.2%

  $ UTRAIN_COMPUTE_FIXTURE="$TESTDIR/fixtures/compute-empty.json" utrain compute list
  KIND  NAME                CORES  POWER  COMPUTE  MEM_USED  MEM_TOTAL  MEM_PCT
  cpu   AMD Ryzen 9 7940HS  16     --     --       8.4       31.0       27.1%
