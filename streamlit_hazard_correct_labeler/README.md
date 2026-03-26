# Streamlit Hazard Correct Labeler

`Sampling_aligned_triplets/**.jsonl` 파일 전체를 한 화면에서 보여주고, 각 항목의 `groundtruth_hazard`(+ 가능하면 `response_hazard`)에 대해 사람은 `true/false`를 선택할 수 있습니다.

기본 동작은 원본 JSONL을 수정하지 않고, 별도 human 라벨 파일(`human_labels/<annotator_id>/...__labels.jsonl`)에 저장합니다.
옵션으로 원본(`hazard_correct`)을 직접 덮어쓸 수도 있습니다.

## 실행

```bash
pip install -r streamlit_hazard_correct_labeler/requirements.txt
streamlit run streamlit_hazard_correct_labeler/app.py
```

## 환경변수(선택)

- `WORKSPACE_ROOT` : 기본값은 `/home/dongwook/EMBGuard_outputs`

