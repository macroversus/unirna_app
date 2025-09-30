def main():
    from argparse import ArgumentParser
    from pathlib import Path
    from rna_app.core.acceptor import infer_acceptor
    from rna_app.core.apa import infer_apa
    from rna_app.core.donor import infer_donor
    from rna_app.core.drugrank import infer_drugrank
    from rna_app.core.lncrna_sublocalization import infer_lncrna_sublocalization
    from rna_app.core.m6a import infer_m6a
    from rna_app.core.pirna import infer_pirna
    from rna_app.core.utr import infer_utr
    from rna_app.core.rna_ss import infer_ss
    from rna_app.core.extract_embedding import extract_embedding
    from rna_app.core.seq_optimization import infer_seq_optimization

    # 当前deeprna存在显存泄漏问题，暂无法通过任何方式解决，因此调用额外的脚本来进行推理
    parser = ArgumentParser()
    parser.add_argument(
        "--in_data", type=str, required=True, help="Input data file path"
    )
    parser.add_argument(
        "--mission",
        type=str,
        required=True,
        help="Mission name",
        choices=[
            "rna_ss",
            "extract_embedding",
            "acceptor",
            "apa",
            "donor",
            "drugrank",
            "lncrna_sublocalization",
            "m6a",
            "pirna",
            "utr",
            "seq_optimization",
        ],
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--model_type", type=str, default="unirna", help="Model type. Only for ss", choices=["unirna", "archiveii"]
    )
    parser.add_argument(
        "--keep_prob", action="store_true", help="Keep probability. Only for ss"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="L16",
        help="Pretrained model name. Only for extract_embedding",
    )
    parser.add_argument(
        "--output_attentions",
        action="store_true",
        help="Output attentions. Only for extract_embedding",
    )
    parser.add_argument(
        "--mutation_ratio",
        type=float,
        default=0.1,
        help="Mutation ratio. Only for seq_optimization",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Optimization iterations. Only for seq_optimization",
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    match args.mission:
        case "rna_ss":
            infer_ss(
                in_data=args.in_data,
                output_dir=args.output_dir,
                model_type=args.model_type,
                return_df=False,
            )
        case "extract_embedding":
            extract_embedding(
                in_data=args.in_data,
                output_dir=args.output_dir,
                pretrained=args.pretrained,
                output_attentions=args.output_attentions,
            )
        case "drugrank":
            # DrugRank needs both CSV and FASTA files
            import os
            fasta_path = os.environ.get("DRUGRANK_FASTA_PATH")
            if not fasta_path:
                raise ValueError("DrugRank requires FASTA path via DRUGRANK_FASTA_PATH environment variable")
            infer_drugrank(
                csv_data=args.in_data,
                fasta_data=fasta_path,
                output_dir=args.output_dir,
                return_df=False,
            )
        case "seq_optimization":
            infer_seq_optimization(
                in_data=args.in_data,
                output_dir=args.output_dir,
                mutation_ratio=args.mutation_ratio,
                iterations=args.iterations,
                return_df=False,
            )
        case _:
            locals()[f"infer_{args.mission}"](
                in_data=args.in_data,
                output_dir=args.output_dir,
                return_df=False,
            )