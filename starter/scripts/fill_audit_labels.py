"""Write the DRAFT audit labels into results/audit_*.csv and summarise them.

The coherence and grounding labels below were assigned by Claude (AI assistant)
on 2026-11-13 after reading each chunk / top-3; they are drafts that the student
must review. Metadata is checked by code against the PDF text and the manifest.
They only apply to the run recorded in results/run_info.json (same chunk IDs).

Run from the repo root:  starter/.venv/Scripts/python starter/scripts/fill_audit_labels.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
from src import run  # noqa: E402

R = ROOT / "results"


def load(name):
    return list(csv.DictReader(open(R / name, encoding="utf-8-sig")))


def save(name, rows):
    with open(R / name, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Coherence, keyed by physical page of wb-taking-stock-2023-08 (positions p11-p21).
COHERENCE_A = {
    11: (1, 'Bắt đầu giữa từ ("al sector") và kết thúc giữa số ("2.9 p"); câu về mất điện ở giữa vẫn trọn ý.'),
    12: (1, "Mở đầu là đuôi câu về dự trữ ngoại hối, kết thúc giữa định nghĩa lạm phát lõi; phần CPI ở giữa đọc được."),
    13: (1, 'Bắt đầu giữa từ ("nts to GDP growth"), kết thúc giữa số ("3.1 p"); các câu dự báo ở giữa hiểu được.'),
    14: (1, 'Bắt đầu bằng mảnh số liệu thiếu chủ thể, kết thúc giữa câu ("targets a"); ý về tỷ lệ giải ngân thấp vẫn hiểu được.'),
    15: (1, 'Mất phần đầu câu ("relaunching the State-Owned"), kết thúc giữa câu ("product base and").'),
    16: (1, 'Bắt đầu giữa câu về sổ đăng ký tài sản, kết thúc giữa từ ("averag"); câu về phân cấp ở giữa trọn ý.'),
    17: (1, 'Bắt đầu giữa từ ("e resilience"), kết thúc giữa từ ("mechanism t"); khuyến nghị ở giữa đọc được.'),
    19: (1, 'Mất chủ thể và con số chính (tăng trưởng GDP 3,7%) do cắt ở "/y)"; đuôi dính tiêu đề mục; câu về xuất khẩu vẫn trọn.'),
    20: (1, 'Bắt đầu giữa từ ("ints"), kết thúc giữa danh sách mặt hàng ("computers"); nội dung chính hiểu được.'),
    21: (1, 'Bắt đầu và kết thúc giữa câu ("external dema"); thiếu chủ thể của câu đầu.'),
}
COHERENCE_B = {
    11: (2, "Bắt đầu và kết thúc ở ranh giới câu; trọn ý về đầu tư và phía sản xuất của GDP."),
    12: (2, "Trọn đoạn về lạm phát, kết thúc bằng câu chủ đề hoàn chỉnh về nới lỏng tiền tệ."),
    13: (2, "Các câu dự báo hoàn chỉnh, tự hiểu được."),
    14: (1, 'Mở bằng "However" phụ thuộc câu trước và kết thúc ở dấu ";" giữa một danh sách biện pháp.'),
    15: (2, "Trọn ý về cải cách DNNN và đa dạng hóa xuất khẩu; ranh giới là cuối câu."),
    16: (2, "Trọn ý về quản lý tài sản công và phân cấp; ranh giới là cuối câu."),
    17: (2, 'Một khuyến nghị hoàn chỉnh; chỉ có ký tự bullet "z" thừa từ PDF.'),
    19: (1, 'Có tiêu đề và đủ số liệu, nhưng câu cuối bị cắt do hết trang và dính nhãn "Figure 1.1".'),
    20: (1, 'Câu cuối ("frequent rolling power outages") bị cắt và dính nhãn hình; phần trước trọn ý.'),
    21: (1, 'Văn bản trang bị trộn cột/chú thích ("2023 24 Sales of goods"), đuôi là chú thích; hiểu được một phần.'),
}
# Grounding of the top-3 for the five test questions. C's top-3 equals B's.
GROUNDING = {
    ("A", "q16"): (2, 'Top-1 (2023 tr.42) chứa đủ câu "26 percent in 1992 to almost 60 percent in 2022".'),
    ("A", "q17"): (1, 'Top-2 nêu điều kiện "35 percent by 2050" nhưng bị cắt ở "by 9–", mất con số 9–11%; có ngữ cảnh, thiếu đáp án.'),
    ("A", "q18"): (2, 'Top-1 (2023 tr.51) có cả vế thừa nhận và vế "no regulatory requirement".'),
    ("A", "q19"): (2, "Top-1 (2025 tr.31) nêu mục tiêu 15% cuối 2024; nhưng top-3 lẫn một chunk báo cáo 2023 (7,8%) dễ gây nhầm."),
    ("A", "q20"): (2, "Top-1 và top-2 đều nêu 6,8% (2025) và 6,5% (2026)."),
    ("B", "q16"): (2, "Top-1 chứa đủ câu và cả so sánh quốc tế."),
    ("B", "q17"): (2, "Top-1 chứa trọn câu: dịch chuyển 35% đến 2050 giảm 9–11% nguồn phát bổ sung."),
    ("B", "q18"): (2, "Top-1 chứa đủ cả hai vế của câu trả lời có điều kiện."),
    ("B", "q19"): (2, "Top-1 nêu mục tiêu 15%; cả top-3 tình cờ đều thuộc báo cáo 2025 dù không lọc."),
    ("B", "q20"): (2, "Top-1 chứa câu dự báo; đầu chunk dính số liệu bảng nhưng không ảnh hưởng."),
}


def main():
    rows = load("audit_coherence.csv")
    for row in rows:
        page = int(row["position"].rsplit(":p", 1)[1])
        labels = COHERENCE_A if row["strategy"] == "A" else COHERENCE_B
        row["coherence_0_2"], row["reason"] = labels[page]
        if row["strategy"] == "C":
            row["reason"] += " (C dùng cùng chunk với B.)"
    save("audit_coherence.csv", rows)

    manifest = json.loads(run.MANIFEST.read_text(encoding="utf-8"))
    pages, _ = run.load_corpus(manifest)
    page_by = {(p.document_id, p.page): p for p in pages}
    url = {d["source_id"]: d["source_path_or_url"] for d in manifest["documents"]}
    traces = {s: {t["id"]: t for t in map(json.loads, open(R / f"trace_{s}.jsonl", encoding="utf-8"))}
              for s in "ABC"}
    rows = load("audit_metadata.csv")
    for row in rows:
        hit = traces[row["strategy"]][row["question_id"]]["hits"][int(row["rank"]) - 1]
        page = page_by[(hit["document_id"], hit["page"])]
        errors = []
        if hit["text"] not in page.text:
            errors.append("chunk text not on that PDF page")
        if hit["source"] != url[hit["document_id"]]:
            errors.append("source != manifest URL")
        for key in ("year", "section", "document_type"):
            if hit["metadata"].get(key) != page.metadata.get(key):
                errors.append(f"{key} mismatch")
        row["metadata_complete_and_correct_0_1"] = 0 if errors else 1
        row["error_note"] = "; ".join(errors) or "source, page, year, section khớp PDF và manifest (đối chiếu bằng code)."
    save("audit_metadata.csv", rows)

    rows = load("audit_grounding.csv")
    for row in rows:
        base = "B" if row["strategy"] == "C" else row["strategy"]
        row["grounding_0_2"], row["reason"] = GROUNDING[(base, row["question_id"])]
        if row["strategy"] == "C":
            row["reason"] += " (Top-3 của C trùng B" + (
                "; filter year=2025 được áp dụng.)" if row["question_id"] == "q19" else ".)")
    save("audit_grounding.csv", rows)

    summary = {}
    coherence, metadata, grounding = (load(f"audit_{n}.csv") for n in ("coherence", "metadata", "grounding"))
    for s in "ABC":
        c = [int(r["coherence_0_2"]) for r in coherence if r["strategy"] == s]
        m = [int(r["metadata_complete_and_correct_0_1"]) for r in metadata if r["strategy"] == s]
        g = [int(r["grounding_0_2"]) for r in grounding if r["strategy"] == s]
        summary[s] = {"coherence_mean": sum(c) / len(c), "coherence_n": len(c),
                      "metadata_correct": f"{sum(m)}/{len(m)}",
                      "grounding_counts_0_1_2": [g.count(0), g.count(1), g.count(2)],
                      "grounding_mean": sum(g) / len(g)}
    summary["_labeler"] = ("Draft labels by Claude (AI assistant), 2026-11-13; metadata checked by code "
                           "against PDF text and manifest. Pending review by the student.")
    (R / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()