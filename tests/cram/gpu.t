  $ . "$TESTDIR/setup.sh"
  $ RID=$(utrain run create --name gpu --image utrain-fake-gpu --compute gpu0 --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait | grep '^status:'
  status:   done

The gpu-check phase runs nvidia-smi inside the container; its output proves the
host GPU is visible from within the container.
  $ utrain run logs "$RID" --phase gpu-check | grep -q NVIDIA && echo "gpu visible"
  gpu visible
