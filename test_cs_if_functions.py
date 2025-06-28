"""
测试新添加的cs_if函数
"""
import polars as pl
from polars_ta.wq.cross_sectional import (
    cs_rank_if, cs_top_bottom_if, cs_qcut_if, cs_scale_if, cs_scale_down_if,
    cs_truncate_if, cs_one_side_if, cs_fill_mean_if, cs_fill_max_if, cs_fill_min_if,
    cs_fill_null_if, cs_fill_except_all_null_if, cs_regression_neut_if, cs_regression_proj_if
)


def test_cs_if_functions():
    """测试cs_if函数的基本功能"""
    
    # 创建测试数据
    df = pl.DataFrame({
        'a': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'b': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'condition': [True, True, False, True, False, True, False, True, False, True, False],
    })
    
    # 测试各种cs_if函数
    result = df.with_columns([
        cs_rank_if(pl.col('condition'), pl.col('a'), True).alias('rank_if_result'),
        
        # 测试cs_top_bottom_if
        cs_top_bottom_if(pl.col('condition'), pl.col('a'), 2).alias('top_bottom_if_result'),
        
        # 测试cs_qcut_if
        cs_qcut_if(pl.col('condition'), pl.col('a'), 4).alias('qcut_if_result'),
        
        # 测试cs_scale_if
        cs_scale_if(pl.col('condition'), pl.col('a'), 1.0).alias('scale_if_result'),
        
        # 测试cs_scale_down_if
        cs_scale_down_if(pl.col('condition'), pl.col('a'), 0).alias('scale_down_if_result'),
        
        # 测试cs_truncate_if
        cs_truncate_if(pl.col('condition'), pl.col('a'), 0.1).alias('truncate_if_result'),
        
        # 测试cs_one_side_if
        cs_one_side_if(pl.col('condition'), pl.col('a'), True).alias('one_side_if_result'),
        
        # 测试cs_fill_mean_if
        cs_fill_mean_if(pl.col('condition'), pl.col('a')).alias('fill_mean_if_result'),
        
        # 测试cs_fill_max_if
        cs_fill_max_if(pl.col('condition'), pl.col('a')).alias('fill_max_if_result'),
        
        # 测试cs_fill_min_if
        cs_fill_min_if(pl.col('condition'), pl.col('a')).alias('fill_min_if_result'),
        
        # 测试cs_fill_null_if
        cs_fill_null_if(pl.col('condition'), pl.col('a'), 0).alias('fill_null_if_result'),
        
        # 测试cs_fill_except_all_null_if
        cs_fill_except_all_null_if(pl.col('condition'), pl.col('a'), 0).alias('fill_except_all_null_if_result'),
        
        # 测试cs_regression_neut_if
        cs_regression_neut_if(pl.col('condition'), pl.col('a'), pl.col('b')).alias('regression_neut_if_result'),
        
        # 测试cs_regression_proj_if
        cs_regression_proj_if(pl.col('condition'), pl.col('a'), pl.col('b')).alias('regression_proj_if_result'),
    ])
    
    print("测试结果:")
    print(result)
    
    # 验证条件筛选是否生效
    print("\n验证条件筛选:")
    print("condition=True的行应该有值，condition=False的行应该为null")
    
    # 检查rank_if_result列
    rank_results = result.select(['condition', 'rank_if_result']).filter(pl.col('condition') == True)
    print(f"condition=True的行数: {len(rank_results)}")
    print(f"其中非null值的行数: {len(rank_results.filter(pl.col('rank_if_result').is_not_null()))}")
    
    rank_results_null = result.select(['condition', 'rank_if_result']).filter(pl.col('condition') == False)
    print(f"condition=False的行数: {len(rank_results_null)}")
    print(f"其中null值的行数: {len(rank_results_null.filter(pl.col('rank_if_result').is_null()))}")


if __name__ == "__main__":
    test_cs_if_functions() 