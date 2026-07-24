  $ . "$TESTDIR/setup.sh"
  $ utrain compute list
  KIND  NAME                 CORES  POWER (W)  COMPUTE  MEM_USED (GB)  MEM_TOTAL (GB)  MEM_USED (%)
  cpu   AMD Ryzen 9 7940HS   16     --         --       8.4            31.0            27.1%
  gpu   NVIDIA RTX 2000 Ada  --     9.5/140.0  8%       0.1            8.0             1.2%

  $ UTRAIN_COMPUTE_FIXTURE="$TESTDIR/fixtures/compute-empty.json" utrain compute list
  KIND  NAME                CORES  POWER (W)  COMPUTE  MEM_USED (GB)  MEM_TOTAL (GB)  MEM_USED (%)
  cpu   AMD Ryzen 9 7940HS  16     --         --       8.4            31.0            27.1%
