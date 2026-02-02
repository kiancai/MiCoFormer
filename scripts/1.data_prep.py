import sys
import os
import logging
from micoformer.utils import build_anndata_from_files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Define project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. 物种丰度表，与 metadata 表的路径位置
    raw_abundance_file = os.path.join(project_root, "data/ResMicroDb_90338/abundance_all_90338.csv")
    raw_metadata_file = os.path.join(project_root, "data/ResMicroDb_90338/Phyloseq_ResMicroDb_metadata_merge_v2_251129.tsv")
    
    # 2. 输出的 anndata 文件路径
    output_file = os.path.join(project_root, "data/processed/microbiome_dataset.h5ad")
    
    # 3. 物种丰度表是否转置
    # 正常情况下，微生物物种丰度表均为 Sample 在列，Taxon/OTU ID 在行
    # 这种情况下，导入 anndata 需要转置，设为 True
    # 如果你的丰度表是 Sample 在行，Taxon/OTU ID 在列，则需要设为 False
    TRANSPOSE = True 

    logger.info("Starting data preparation pipeline...")
    logger.info(f"Project root: {project_root}")
    
    try:
        build_anndata_from_files(
            abundance_path=raw_abundance_file,
            metadata_path=raw_metadata_file,
            output_path=output_file,
            transpose_abundance=TRANSPOSE,
            sparse_format=True
        )
    except Exception as e:
        logger.error(f"\n❌ Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()