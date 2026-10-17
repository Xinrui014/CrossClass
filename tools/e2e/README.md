# tools/e2e — end-to-end evaluation scripts (2026-10-15..17)

Copied from the session scratchpad so they persist; most were run on the L40 as /tmp/xinrui_<name> with ~/venvs/gdino_fewshot
and read /tmp/xinrui_e2e (e2e boards, OCR boxes, links, PARSeq features) — rebuild those inputs with e2e_build*.py /
text_features.py boxes / e2e_infer_tf.py when /tmp is gone. Key entry points: e2e_after.sh (check one intern02 checkpoint end to end),
e2e_attrib.py (error attribution), e2e_score_conf.py (box score modes), full_metrics.py (main table), ../score_b.py and
../e2e_infer_b.py (proposal B). llm_naming/: Claude's and the blind agent's per-board prefix decisions + their inputs.
Not committed; paths inside the scripts still point to the scratchpad / /tmp in places.
