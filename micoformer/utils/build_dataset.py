import os
import logging
import pandas as pd
import anndata
from scipy import sparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_sep(file_path: str) -> str:
    """根据文件后缀推断分隔符"""
    if file_path.endswith('.tsv'):
        return '\t'
    elif file_path.endswith('.csv'):
        return ','
    else:
        raise ValueError(f"Unsupported file extension for '{file_path}'. Only .csv and .tsv are supported.")

def build_anndata_from_files(
    abundance_path: str,
    metadata_path: str,
    output_path: str,
    transpose_abundance: bool = True,
    sparse_format: bool = True
) -> None:
    """
    将原始丰度表和元数据表合并并转换为 AnnData (.h5ad) 格式。
    参数:
        abundance_path (str): 丰度表文件路径 (CSV/TSV)。
        metadata_path (str): 元数据表文件路径 (CSV/TSV)。
        output_path (str): 输出 .h5ad 文件的保存路径。
        transpose_abundance (bool): 是否需要转置丰度表。
                                    - True (默认): 假设输入为 [行=Taxa, 列=Samples]，转置为 [Samples, Taxa]。
                                    - False: 假设输入已经是 [行=Samples, 列=Taxa]。
        sparse_format (bool): 是否将数据矩阵压缩为稀疏矩阵 (CSR)。强烈建议 True，节省大量内存。
    """
    
    # 自动推断分隔符
    try:
        abundance_sep = _get_sep(abundance_path)
        metadata_sep = _get_sep(metadata_path)
    except ValueError as e:
        logger.error(str(e))
        raise

    # --- 1. 读取丰度表 (OTU Table / ASV Table) ---
    logger.info(f"Reading abundance table: {abundance_path}")
    try:
        # index_col=0 表示第一列是索引 (通常是 Taxon ID 或 OTU ID)
        df_abund = pd.read_csv(abundance_path, sep=abundance_sep, index_col=0)
        logger.info(f"Original abundance table shape: {df_abund.shape} (Rows x Cols)")
    except Exception as e:
        logger.error(f"Failed to read abundance table. Please check path or format. Error: {e}")
        raise

    # --- 2. 转置与维度确认 ---
    # AnnData 要求：Rows=样本(Samples), Cols=特征(Taxa)
    if transpose_abundance:
        logger.info("Transposing abundance table (Rows=Taxa -> Rows=Samples)...")
        df_abund = df_abund.T
    
    # 此时 df_abund 的行应该是样本，列应该是物种
    n_samples_abund, n_taxa = df_abund.shape
    logger.info(f"Abundance table loaded: {n_samples_abund} samples x {n_taxa} taxa/features.")

    # --- 3. 读取 Metadata ---
    logger.info(f"Reading metadata: {metadata_path}")
    try:
        # 读取 Metadata，不指定 index_col，先读进来再找 'Run' 列
        df_meta = pd.read_csv(metadata_path, sep=metadata_sep, low_memory=False)
        
        # 尝试查找 'Run' 列作为索引，如果没有则尝试第一列
        if 'Run' in df_meta.columns:
            logger.info("Found 'Run' column, using it as sample index.")
            df_meta = df_meta.set_index('Run')
        else:
            logger.warning("'Run' column not found in metadata. Using the first column as sample index.")
            df_meta = df_meta.set_index(df_meta.columns[0])
            
        n_samples_meta, n_features_meta = df_meta.shape
        logger.info(f"Metadata loaded: {n_samples_meta} samples x {n_features_meta} metadata features.")

            # --- 5. 交互式类型检查与自动修复 (Interactive Type Check & Auto-fix) ---
        logger.info("Checking metadata types and fixing mixed-type issues for AnnData compatibility...")
        
        # 1. 对于 object 类型的列（通常是字符串），如果有 NaN，必须填充为空字符串，否则 AnnData 写入会报错。
        # 2. 对于 numeric 类型的列（float/int），NaN 是允许的，AnnData 可以处理。
        
        fixed_columns = []
        
        for col in df_meta.columns:

            dtype = df_meta[col].dtype
            has_nan = df_meta[col].isna().any()
            
            if pd.api.types.is_object_dtype(dtype):
                if has_nan:
                    df_meta[col] = df_meta[col].fillna("").astype(str)
                    fixed_columns.append(col)
                else:
                    # 即使没有 NaN，也强制转为 string 以防万一
                    df_meta[col] = df_meta[col].astype(str)
                    
        if fixed_columns:
            logger.warning(f"Auto-fixed {len(fixed_columns)} columns containing NaNs (filled with empty string): {', '.join(fixed_columns)}")
        else:
            logger.info("No object columns with NaNs found.")

        logger.info("Metadata type cleaning complete.")
        

    except Exception as e:
        logger.error(f"Failed to read metadata. Error: {e}")
        raise

    # --- 4. 数据对齐 ---
    # 找出两个表中都存在的样本 ID（取交集）
    # df_abund.index 现在应该是样本 ID
    # df_meta.index 现在也应该是样本 ID
    common_samples = df_abund.index.intersection(df_meta.index)
    n_common = len(common_samples)
    
    if n_common == 0:
        error_msg = (
            "Error: No overlapping Sample IDs found between abundance table and metadata!\n"
            f"Abundance table Sample ID example: {df_abund.index[:5].tolist()}\n"
            f"Metadata Sample ID example: {df_meta.index[:5].tolist()}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Data alignment successful: Found {n_common} common samples.")
    
    # 如果样本数不一致，打印警告
    if n_common < df_abund.shape[0] or n_common < df_meta.shape[0]:
        logger.warning(f"Discarded some unmatched samples: Abundance table remaining {df_abund.shape[0]-n_common}, Metadata remaining {df_meta.shape[0]-n_common}")

    # 按照共同 ID 筛选并排序，确保一一对应
    df_abund = df_abund.loc[common_samples]
    df_meta = df_meta.loc[common_samples]

    # --- 5. 构建 AnnData 对象 ---
    logger.info("Building AnnData object...")
    
    # 检查是否有非数值数据混入丰度表
    try:
        X_values = df_abund.values.astype('float32')
    except ValueError:
        logger.error("Abundance table contains non-numeric characters, cannot convert! Please check for comment lines or incorrect columns.")
        raise

    # 构建对象
    # obs: 观测（样本）的元数据
    # var: 变量（Taxa）的元数据（这里暂时只有 Taxon ID，如果 Taxon 有其他属性如 NCBI ID 也可以加进去）
    adata = anndata.AnnData(X=X_values, obs=df_meta)
    
    # 将 Taxon ID 设置为 var_names
    adata.var_names = df_abund.columns
    # 将 Sample ID 设置为 obs_names
    adata.obs_names = df_abund.index

    # --- 6. 稀疏化处理 (Sparsity) ---
    if sparse_format:
        logger.info("Converting data matrix to CSR sparse format (saving memory)...")
        adata.X = sparse.csr_matrix(adata.X)

    # --- 7. 保存文件 ---
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Saving to: {output_path}")
    adata.write(output_path, compression="gzip") # 使用 gzip 压缩进一步减小体积
    
    logger.info("Processing complete!")
    
    # --- 8. 最终报告 (Final Report) ---
    # Use print instead of logger to avoid timestamp prefix for table formatting
    print("="*50)
    print("【Final AnnData Summary】")
    print(f"Total Samples: {adata.n_obs}")
    print(f"Total Features: {adata.n_vars}")
    print(f"Total Metadata Columns: {len(adata.obs.columns)}")
    print("-" * 50)
    print(f"{'Metadata Column':<30} | {'Final Dtype in AnnData'}")
    print("-" * 50)
    
    for col in adata.obs.columns:
        dtype_str = str(adata.obs[col].dtype)
        print(f"{col:<30} | {dtype_str}")
        
    print("="*50)
    logger.info(f"AnnData object saved to: {output_path}")