# Experience-Shaped Medium v0 — 公開再現手順

この実験は2026-10-18の**有限6×6局所物理媒体の因果効果**を検証する。知性・任意自由回路・LLMによる難問解決を再現するものではない。

## 環境と単独smoke

Python 3.11、NumPy。リポジトリのルートから実行する。

~~~bash
git clone https://github.com/tacky0307/SphereBrain-Public.git
cd SphereBrain-Public
python -m pip install numpy
python -m unittest -v research.experiments.experience_shaped_medium_v0.test_contract
python -m research.experiments.experience_shaped_medium_v0.implementation --seed 332001 --out out/medium332/smoke/seed_332001.json
python -m research.experiments.experience_shaped_medium_v0.aggregate --raw out/medium332/smoke --cohort smoke --out out/medium332/smoke/audit.json
~~~

smokeは**非正式の予備実験**。C0がPASSになり、正式6例用ゲートには必要個体数不足のため PARTIAL と出るのが正常。

## 診断6seedと公式holdout6seedの再計算

~~~bash
mkdir -p out/medium332/diagnostic out/medium332/holdout
for seed in 332101 332102 332103 332104 332105 332106; do
  python -m research.experiments.experience_shaped_medium_v0.implementation --seed "$seed" --out "out/medium332/diagnostic/seed_$seed.json"
done
python -m research.experiments.experience_shaped_medium_v0.aggregate --raw out/medium332/diagnostic --cohort diagnostic --out out/medium332/diagnostic/audit.json
for seed in 332201 332202 332203 332204 332205 332206; do
  python -m research.experiments.experience_shaped_medium_v0.implementation --seed "$seed" --out "out/medium332/holdout/seed_$seed.json"
done
python -m research.experiments.experience_shaped_medium_v0.aggregate --raw out/medium332/holdout --cohort holdout --out out/medium332/holdout/audit.json
~~~

Windows PowerShell版（holdoutのみ）：

~~~powershell
New-Item -ItemType Directory -Force -Path out/medium332/holdout | Out-Null
332201..332206 | ForEach-Object {
  python -m research.experiments.experience_shaped_medium_v0.implementation --seed $_ --out "out/medium332/holdout/seed_$_.json"
}
python -m research.experiments.experience_shaped_medium_v0.aggregate --raw out/medium332/holdout --cohort holdout --out out/medium332/holdout/audit.json
~~~

- audit.json の c0 は **PASS**、verdict は **MEDIUM_TRACE_CAUSAL_EFFECT_SUPPORTED_IN_FIXED_GRID**。
- means と individual の M0〜M7 を[RESULTS_JA.md](RESULTS_JA.md)の公式値に照合する。閾値は変更しない。
- 生成された raw JSON はA/Bの物理入力振幅、媒体伝導率、全介入条件、3つの未来probe各24時刻×36セルの全活動を収録する。
- aggregate.py は implementation.py をimportせず、**別実装の物理計算**で各tickと全未来probeの状態・刺激・保存帳簿・集計を再計算する。
- これは人工法則の一貫性監査であり、実際の自然界で同じ結果が出るという主張ではない。

## 公開のGitHub Actionsで再現

[公開用GitHub Actionsワークフロー](../../../.github/workflows/experience-shaped-medium-332-reproduction.yml)が、契約検査、6診断seed、6holdout seed、独立監査を実行し、rawと独立レポートを成果物として残す。

元の科学的事前登録は、実装より先に private 研究側の commit **61d7203e9435afcf9057a8039395f6fade95d733**で固定。元の未使用正式6seedはprivateリポジトリで一回実施し、その結果をここに公開した。**公開後のCI再実行は既知の結果の再現であり、新しい未使用holdoutとは呼ばない。**

数値係数・形状・seed・閾値を変更した場合は別研究とし、[PROTOCOL.md](PROTOCOL.md)と過去の成功/失敗を上書きしないこと。公開版Unified Experience Core v2.0とは独立した再現パッケージであり、旧版の仕様・結果を変更しない。
