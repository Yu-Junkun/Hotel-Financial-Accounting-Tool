#./BaiduSyncdisk/Python_Project/venv/Scripts/python.exe
# -*- coding: utf-8 -*-
'''
@File    :   gm_streamlit.py
@Time    :   2025/07/25 17:46:59
@Author  :   Junkun Yu
@Version :   1.0
@Desc    :   None
'''

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
# import chardet

# 设置页面配置
st.set_page_config(layout="wide")

# # 检测文件编码
# def detect_encoding(file_path):
#     with open(file_path, 'rb') as f:
#         result = chardet.detect(f.read(10000))  # 读取前10KB来检测编码
#     return result['encoding']

# 检测文件编码 - 修改为直接处理文件对象
# def detect_encoding(file_object):
#     """检测文件对象的编码"""
#     # 读取前10KB来检测编码
#     raw_data = file_object.read(10240)
#     # 重置文件指针，以便后续读取
#     file_object.seek(0)
#     result = chardet.detect(raw_data)
#     return result['encoding']

def reset_state():
    st.session_state.gm_pms_df = None
    st.session_state.gm_pos_df = None
    st.session_state.gm_compare_result = None
    
    st.session_state.fg_pms_df = None
    st.session_state.fg_pos_df = None
    st.session_state.fg_compare_result = None
    
    st.session_state.yn_pms_df = None
    st.session_state.yn_pos_df = None
    st.session_state.yn_ota_df = None
    st.session_state.yn_compare_result = None
    
    st.session_state.lz_pms_df = None
    st.session_state.lz_pos_df = None
    st.session_state.lz_compare_result = None
    
# 数据处理函数
def gm_fg_yn_process_pms_data(pms_df, pms_calc_list):
    """处理PMS数据，计算核算金额"""
    pms_df['日期'] = pd.to_datetime(pms_df['日期'], errors='coerce')
    
    # 检查无效日期
    invalid_dates = pms_df[pms_df['日期'].isna()]
    if not invalid_dates.empty:
        st.warning(f"警告：PMS文件中存在{len(invalid_dates)}行无效日期，已标记为NaT")
    
    # 计算核算金额
    pms_df['核算金额'] = pms_df.reindex(columns=pms_calc_list).sum(axis=1).round(2)
    return pms_df

def yn_process_pms_data_for_ota(pms_df):
    """处理PMS数据，提取核对单号并计算核算金额"""
    # 1. 处理单号列（增加类型检查）
    def extract_order_number(x):
        if pd.notna(x) and isinstance(x, str):  # 检查是否为非空字符串
            if "：" in x:
                return x.split("：", 1)[1].strip()
        return None  # 非字符串或不含冒号返回None
    
    pms_df["核对单号"] = pms_df["单号"].apply(extract_order_number)
    pms_df["核对单号"] = pms_df["核对单号"].fillna('').astype(str)
    pms_df = pms_df[['核对单号', '消费']]
    pms_df = pms_df[pms_df['核对单号'] != '']
    return pms_df

def lz_process_pms_data_for_ota(pms_df, convert_df):
    # 确保日期列是datetime类型
    pms_df['入住时间'] = pd.to_datetime(pms_df['入住时间'])
    pms_df['退房时间'] = pd.to_datetime(pms_df['退房时间'])
    pms_df['单号'] = pms_df['单号'].fillna('').astype(str)
    # convert_df['单号'] = convert_df['单号'].fillna('').astype(str)
    # convert_df['OTA单号'] = convert_df['OTA单号'].fillna('').astype(str)
    # 按单号分组，同时聚合消费总和、最早入住时间和最晚退房时间
    pms_grouped = pms_df.groupby('单号').agg({
        '消费': 'sum',
        '入住时间': 'min',  # 最早入住时间
        '退房时间': 'max'   # 最晚退房时间
    }).reset_index()
    
    # 重命名列
    pms_grouped.rename(columns={
        '消费': 'PMS同单号消费总和',
        '入住时间': '最早入住时间',
        '退房时间': '最晚退房时间'
    }, inplace=True)
    
    # 合并数据
    pms_comverted_df = pd.merge(
        pms_grouped[['单号', 'PMS同单号消费总和', '最早入住时间', '最晚退房时间']],
        convert_df[['单号', 'OTA单号']],
        on='单号',  # 左右列名相同时可以简化
        how='left'
    )
    
    return pms_comverted_df

def lz_process_ota_data(ota_df):
    # ota_df['订单号'] = ota_df['订单号'].fillna('').astype(str)
    ota_df = ota_df.groupby('订单号')['结算价'].sum().reset_index()
    ota_df.rename(columns={'结算价': 'OTA同订单号结算价总和'}, inplace=True)
    return ota_df

def lz_compare_data(pms_df, ota_df):
    """比对PMS和POS数据，计算差异"""
    # 检查输入DataFrame是否有效
    ota_df['订单号'] = ota_df['订单号'].fillna('').astype(str)
    if pms_df.empty or ota_df.empty:
        st.error("无法比对数据：PMS或OTA数据为空")
        return pd.DataFrame()
    
    # 检查必要的列是否存在
    if 'PMS同单号消费总和' not in pms_df.columns:
        st.error("PMS数据缺少'PMS同单号消费总和'列")
        return pd.DataFrame()
    
    if 'OTA同订单号结算价总和' not in ota_df.columns:
        st.error("OTA数据缺少'OTA同订单号结算价总和'列")
        return pd.DataFrame()
    
    # 合并数据
    compare_result = pd.merge(
        pms_df[['单号', 'OTA单号', 'PMS同单号消费总和', '最早入住时间', '最晚退房时间']],
        ota_df[['订单号', 'OTA同订单号结算价总和']],
        left_on='OTA单号',
        right_on='订单号',
        how='left' 
    )
    
    # 填充缺失值
    compare_result['OTA同订单号结算价总和'] = compare_result['OTA同订单号结算价总和'].fillna(0)
    
    # 计算差异
    compare_result['PMS同单号消费总和'] = compare_result['PMS同单号消费总和'].round(2)
    compare_result['OTA同订单号结算价总和'] = compare_result['OTA同订单号结算价总和'].round(2)
    compare_result['差异'] = compare_result['PMS同单号消费总和'] - compare_result['OTA同订单号结算价总和']
    compare_result['差异'] = compare_result['差异'].round(2)
    
    # 更改日期格式为%Y/%d/%m %H:%M:%S 入住时间和退房时间
    compare_result['最早入住时间'] = compare_result['最早入住时间'].dt.strftime('%Y/%m/%d %H:%M:%S')
    compare_result['最晚退房时间'] = compare_result['最晚退房时间'].dt.strftime('%Y/%m/%d %H:%M:%S')
    compare_result = compare_result[['单号', 'OTA单号', '订单号', '最早入住时间', '最晚退房时间', 'PMS同单号消费总和', 'OTA同订单号结算价总和', '差异']]
    compare_result.rename(columns={'单号': 'PMS_内部单号', 'OTA单号': 'PMS_OTA单号', '订单号': 'OTA单号'}, inplace=True)
    
    return compare_result

def gm_yn_process_pos_data(pos_df):
    """处理POS数据，计算核算日期和每日总和"""
    # 检查必要的列是否存在
    required_columns = ['交易时间', '交易金额']
    missing_columns = [col for col in required_columns if col not in pos_df.columns]
    
    if missing_columns:
        st.error(f"POS数据缺少必要的列: {', '.join(missing_columns)}")
        return pd.DataFrame()  # 返回空DataFrame
    
    pos_df['交易时间'] = pd.to_datetime(pos_df['交易时间'], errors='coerce')
    
    # 检查无效时间
    invalid_times = pos_df[pos_df['交易时间'].isna()]
    if not invalid_times.empty:
        st.warning(f"警告：POS文件中存在{len(invalid_times)}行无效交易时间，已标记为NaT")
    
    # 计算核算日期（04:00-次日04:00为一个周期）
    pos_df['核算日期'] = pos_df['交易时间'].apply(lambda x: x.date() if x.hour >= 4 else (x - pd.Timedelta(days=1)).date())
    pos_df['核算日期'] = pd.to_datetime(pos_df['核算日期'])
    
    # 按核算日期汇总交易金额
    pos_df_daily = pos_df.groupby('核算日期')['交易金额'].sum().reset_index()
    pos_df_daily.rename(columns={'交易金额': '表二时段交易总和'}, inplace=True)
    return pos_df_daily

def fg_lz_process_pos_data(pos_df, default_hour=4):
    """处理POS数据，计算核算日期和每日总和"""
    # 检查必要的列是否存在
    required_columns = ['交易时间', '交易金额']
    missing_columns = [col for col in required_columns if col not in pos_df.columns]
    
    if missing_columns:
        st.error(f"POS数据缺少必要的列: {', '.join(missing_columns)}")
        return pd.DataFrame()  # 返回空DataFrame
    
    pos_df['交易时间'] = pd.to_datetime(pos_df['交易时间'], errors='coerce')
    
    # 检查无效时间
    invalid_times = pos_df[pos_df['交易时间'].isna()]
    if not invalid_times.empty:
        st.warning(f"警告：POS文件中存在{len(invalid_times)}行无效交易时间，已标记为NaT")
    
    # 计算核算日期（04:00-次日04:00为一个周期）
    pos_df['核算日期'] = pos_df['交易时间'].apply(lambda x: x.date() if x.hour >= default_hour else (x - pd.Timedelta(days=1)).date())
    pos_df['核算日期'] = pd.to_datetime(pos_df['核算日期'])
    # 筛选交易类型不为“预授权”“预授权撤销”“预授权调整”
    pos_df = pos_df[pos_df['交易类型'] != '预授权']
    pos_df = pos_df[pos_df['交易类型'] != '预授权撤销']
    
    # 按核算日期汇总交易金额
    pos_df_daily = pos_df.groupby('核算日期')['交易金额'].sum().reset_index()
    pos_df_daily.rename(columns={'交易金额': '表二时段交易总和'}, inplace=True)
      
    return pos_df_daily

def gm_fg_yn_compare_data(pms_df, pos_df_daily):
    """比对PMS和POS数据，计算差异"""
    # 检查输入DataFrame是否有效
    if pms_df.empty or pos_df_daily.empty:
        st.error("无法比对数据：PMS或POS数据为空")
        return pd.DataFrame()
    
    # 检查必要的列是否存在
    if '核算金额' not in pms_df.columns:
        st.error("PMS数据缺少'核算金额'列")
        return pd.DataFrame()
    
    if '表二时段交易总和' not in pos_df_daily.columns:
        st.error("POS数据缺少'表二时段交易总和'列")
        return pd.DataFrame()
    
    # 合并数据
    compare_result = pd.merge(
        pms_df[['日期', '核算金额']],
        pos_df_daily,
        left_on='日期',
        right_on='核算日期',
        how='left'
    )
    
    # 填充缺失值
    compare_result['表二时段交易总和'] = compare_result['表二时段交易总和'].fillna(0)
    
    # 计算差异
    compare_result['核算金额'] = compare_result['核算金额'].round(2)
    compare_result['表二时段交易总和'] = compare_result['表二时段交易总和'].round(2)
    compare_result['差异'] = compare_result['表二时段交易总和'] - compare_result['核算金额']
    compare_result['差异'] = compare_result['差异'].round(2)
    
    # 更改日期格式为%Y/%d/%m
    compare_result['日期'] = compare_result['日期'].dt.strftime('%Y/%m/%d')
    compare_result = compare_result[['日期', '核算金额', '表二时段交易总和', '差异']]
    
    return compare_result

def yn_process_ota_data(ota_df):
    """处理ota数据"""
    # 检查必要的列是否存在
    required_columns = ['订单号', '分成金额']
    missing_columns = [col for col in required_columns if col not in ota_df.columns]
    
    if missing_columns:
        st.error(f"ota数据缺少必要的列: {', '.join(missing_columns)}")
        return pd.DataFrame()  # 返回空DataFrame
    ota_df['订单号'] = ota_df['订单号'].fillna('').astype(str)

    return ota_df

def yn_compare_data(pms_df, ota_df):
    """比对PMS和POS数据，计算差异"""
    # 检查输入DataFrame是否有效
    if pms_df.empty or ota_df.empty:
        st.error("无法比对数据：PMS或ota数据为空")
        return pd.DataFrame()
    
    # 检查必要的列是否存在
    if '消费' not in pms_df.columns:
        st.error("PMS数据缺少'消费'列")
        return pd.DataFrame()
    # 检查必要的列是否存在
    if '核对单号' not in pms_df.columns:
        st.error("PMS数据缺少'核对单号'列")
        return pd.DataFrame()
    
    if '分成金额' not in ota_df.columns:
        st.error("ota数据缺少'分成金额'列")
        return pd.DataFrame()
    
    # 合并数据
    compare_result = pd.merge(
        pms_df[['消费', '核对单号']],
        ota_df,
        left_on='核对单号',
        right_on='订单号',
        how='left'
    )
    
    # 填充缺失值
    compare_result['分成金额'] = compare_result['分成金额'].fillna(0)
    compare_result['消费'] = compare_result['消费'].fillna(0)
    
    # 计算差异
    compare_result['分成金额'] = compare_result['分成金额'].round(2)
    compare_result['消费'] = compare_result['消费'].round(2)
    compare_result['差异'] = compare_result['消费'] - compare_result['分成金额']
    compare_result['差异'] = compare_result['差异'].round(2)
    
    # 更改日期格式为%Y/%d/%m
    compare_result = compare_result[['核对单号', '消费', '订单号', '分成金额', '差异']]
    
    return compare_result

def gm_hotel_pms_pos_check():
    st.write('格曼酒店PMS与POS对账系统')
    
    # 初始化会话状态
    if 'gm_pms_df' not in st.session_state:
        st.session_state.gm_pms_df = None
    if 'gm_pos_df' not in st.session_state:
        st.session_state.gm_pos_df = None
    if 'gm_compare_result' not in st.session_state:
        st.session_state.gm_compare_result = None
    if 'gm_file_saved' not in st.session_state:
        st.session_state.gm_file_saved = False
    if 'gm_file_name' not in st.session_state:
        st.session_state.gm_file_name = None
    
    # 文件上传
    col1, col2 = st.columns(2)
    
    with col1:
        gm_upload_file_pms = st.file_uploader("上传PMS文件", type=["xlsx"], key='gm_pms')
        if gm_upload_file_pms is not None:
            st.session_state.gm_pms_df = pd.read_excel(gm_upload_file_pms)[:-1]
            st.success('PMS文件已成功加载')
            st.write('PMS文件前3行预览:')
            st.dataframe(st.session_state.gm_pms_df.head(3))
    
    with col2:
        gm_upload_file_pos = st.file_uploader("上传POS文件", type=["csv"])
        if gm_upload_file_pos is not None:
            # encoding = detect_encoding(upload_file_pos)
            # st.write(encoding)
            st.session_state.gm_pos_df = pd.read_csv(gm_upload_file_pos, header=3, encoding='gbk')
            st.success('POS文件已成功加载')
            st.write('POS文件前3行预览:')
            st.dataframe(st.session_state.gm_pos_df.head(3))
    
    # 对账按钮
    if st.button('开始对账'):
        # 检查文件是否都已上传
        if st.session_state.gm_pms_df is None or st.session_state.gm_pos_df is None:
            st.error('请上传PMS和POS两个文件')
            return
        
        # 数据处理
        with st.spinner('正在处理数据...'):
            pms_calc_list = ['银联卡', '微信支付', '支付宝支付', '支付宝预授', '微信押金', '支付宝押金']
            
            # 处理PMS数据
            processed_pms = gm_fg_yn_process_pms_data(st.session_state.gm_pms_df, pms_calc_list)

            processed_pos = gm_yn_process_pos_data(st.session_state.gm_pos_df)
            
            # 比对数据
            st.session_state.gm_compare_result = gm_fg_yn_compare_data(processed_pms, processed_pos)

            
            # 重置保存状态
            st.session_state.gm_file_saved = False

            
            st.success('对账完成！')
    
    # 显示对账结果（如果有）
    if st.session_state.gm_compare_result is not None:

        st.subheader('对账结果')
        
        # 结果表格
        st.dataframe(
            st.session_state.gm_compare_result[['日期', '核算金额', '表二时段交易总和', '差异']],

            use_container_width=True
        )
        
        total_days = len(st.session_state.gm_compare_result)
        matched_days = len(st.session_state.gm_compare_result[st.session_state.gm_compare_result['差异'] == 0])
        matched_days = len(st.session_state.gm_compare_result[st.session_state.gm_compare_result['差异'] == 0])
        unmatched_days = total_days - matched_days
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总比对天数", total_days)
        col2.metric("匹配天数", matched_days)
        col3.metric("未匹配天数", unmatched_days)
        
        # # 差异图表
        # if unmatched_days > 0:
        #     st.subheader('差异分析')
        #     fig = st.session_state.gm_compare_result[st.session_state.gm_compare_result['差异'] != 0].plot(
        #         x='日期', 
        #         y='差异', 
        #         kind='bar', 
        #         title='每日差异金额',
        #         figsize=(10, 6)
        #     )
        #     st.pyplot(fig.figure)
        
        # 保存和下载按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('保存对账结果'):
                if st.session_state.gm_compare_result is not None:
                    # 生成文件名并保存文件
                    st.session_state.gm_file_name = f'格曼PMS_POS对账结果_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx'
                    st.session_state.gm_compare_result.to_excel(st.session_state.gm_file_name, index=False)
                    st.session_state.gm_file_saved = True
                    st.success('对账结果已保存')
        
        with col2:
            if st.session_state.gm_file_saved and st.session_state.gm_file_name:
                # 提供下载链接
                with open(st.session_state.gm_file_name, 'rb') as f:
                    st.download_button(
                        label="下载格曼对账结果",
                        data=f,
                        file_name=st.session_state.gm_file_name,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                    
def fg_hotel_pms_pos_check():
    st.write('方格酒店PMS与POS对账系统')
    
    # 初始化会话状态
    if 'fg_pms_df' not in st.session_state:
        st.session_state.fg_pms_df = None
    if 'fg_pos_df' not in st.session_state:
        st.session_state.fg_pos_df = None
    if 'fg_compare_result' not in st.session_state:
        st.session_state.fg_compare_result = None
    if 'fg_file_saved' not in st.session_state:
        st.session_state.fg_file_saved = False
    if 'fg_file_name' not in st.session_state:
        st.session_state.fg_file_name = None
    
    # 文件上传
    col1, col2 = st.columns(2)
    
    with col1:
        fg_upload_file_pms = st.file_uploader("上传PMS文件", type=["xlsx"], key='fg_pms')
        if fg_upload_file_pms is not None:
            st.session_state.fg_pms_df = pd.read_excel(fg_upload_file_pms)[:-1]
            st.success('PMS文件已成功加载')
            st.write('PMS文件前3行预览:')
            st.dataframe(st.session_state.fg_pms_df.head(3))
    
    with col2:
        fg_upload_file_pos = st.file_uploader("上传POS文件", type=["xlsx"])
        if fg_upload_file_pos is not None:
            # encoding = detect_encoding(upload_file_pos)
            # st.write(encoding)
            st.session_state.fg_pos_df = pd.read_excel(fg_upload_file_pos)
            st.success('POS文件已成功加载')
            st.write('POS文件前3行预览:')
            st.dataframe(st.session_state.fg_pos_df.head(3))
    
    # 对账按钮
    if st.button('开始对账'):
        # 检查文件是否都已上传
        if st.session_state.fg_pms_df is None or st.session_state.fg_pos_df is None:
            st.error('请上传PMS和POS两个文件')
            return
        
        # 数据处理
        with st.spinner('正在处理数据...'):
            pms_calc_list = ['银联卡', '微信支付', '支付宝支付', '支付宝预授', '微信押金', '支付宝押金']
            
            # 处理PMS数据
            processed_pms = gm_fg_yn_process_pms_data(st.session_state.fg_pms_df, pms_calc_list)
            
            # 处理POS数据
            processed_pos = fg_lz_process_pos_data(st.session_state.fg_pos_df)
            
            # 比对数据
            st.session_state.fg_compare_result = gm_fg_yn_compare_data(processed_pms, processed_pos)
            
            # 重置保存状态
            st.session_state.fg_file_saved = False
            
            st.success('对账完成！')
    
    # 显示对账结果（如果有）
    if st.session_state.fg_compare_result is not None:
        st.subheader('对账结果')
        
        # 结果表格
        st.dataframe(
            st.session_state.fg_compare_result[['日期', '核算金额', '表二时段交易总和', '差异']],
            use_container_width=True
        )
        
        # 差异分析
        total_days = len(st.session_state.fg_compare_result)
        matched_days = len(st.session_state.fg_compare_result[st.session_state.fg_compare_result['差异'] == 0])
        unmatched_days = total_days - matched_days
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总比对天数", total_days)
        col2.metric("匹配天数", matched_days)
        col3.metric("未匹配天数", unmatched_days)
        
        # # 差异图表
        # if unmatched_days > 0:
        #     st.subheader('差异分析')
        #     fig = st.session_state.fg_compare_result[st.session_state.fg_compare_result['差异'] != 0].plot(
        #         x='日期', 
        #         y='差异', 
        #         kind='bar', 
        #         title='每日差异金额',
        #         figsize=(10, 6)
        #     )
        #     st.pyplot(fig.figure)
        
        # 保存和下载按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('保存对账结果'):
                if st.session_state.fg_compare_result is not None:
                    # 生成文件名并保存文件
                    st.session_state.fg_file_name = f'方格PMS_POS对账结果_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx'
                    st.session_state.fg_compare_result.to_excel(st.session_state.fg_file_name, index=False)
                    st.session_state.fg_file_saved = True
                    st.success('对账结果已保存')
        
        with col2:
            if st.session_state.fg_file_saved and st.session_state.fg_file_name:
                # 提供下载链接
                with open(st.session_state.fg_file_name, 'rb') as f:
                    st.download_button(
                        label="下载方格对账结果",
                        data=f,
                        file_name=st.session_state.fg_file_name,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

def yn_hotel_pms_ota_check():
    st.write('雅南酒店PMS与OTA对账系统')
    
    # 初始化会话状态
    if 'yn_pms_df' not in st.session_state:
        st.session_state.yn_pms_df = None
    if 'yn_ota_df' not in st.session_state:
        st.session_state.yn_ota_df = None
    if 'yn_compare_result' not in st.session_state:
        st.session_state.yn_compare_result = None
    if 'yn_file_saved' not in st.session_state:
        st.session_state.yn_file_saved = False
    if 'yn_file_name' not in st.session_state:
        st.session_state.yn_file_name = None
    
    # 文件上传
    col1, col2 = st.columns(2)
    
    with col1:
        yn_upload_file_pms = st.file_uploader("上传PMS文件，请注意将xls文档另存为xlsx", type=["xlsx"], key='yn_pms_ota')
        if yn_upload_file_pms is not None:
            st.session_state.yn_pms_df = pd.read_excel(yn_upload_file_pms)
            st.success('PMS文件已成功加载')
            st.write('PMS文件前3行预览:')
            st.dataframe(st.session_state.yn_pms_df.head(3))
    
    with col2:
        yn_upload_files_ota = st.file_uploader(
            label="上传ota文件",
            type=["xlsx"],
            accept_multiple_files=True  # 允许多文件上传
        )
        
        if yn_upload_files_ota:  # 当有文件上传时
            dfs = []  # 存储所有工作表的数据
            error_messages = []  # 存储错误信息
            st.write(f"正在处理 {len(yn_upload_files_ota)} 个文件")
            for file in yn_upload_files_ota:
                file_name = file.name
                
                
                try:
                    # 1. 用ExcelFile打开文件（高效读取多工作表）
                    excel_file = pd.ExcelFile(file, engine='openpyxl')  # .xlsx需用openpyxl引擎
                    all_sheets = excel_file.sheet_names  # 获取所有工作表名称
                    
                    # 2. 检查并读取"美团"和"携程"工作表
                    for sheet in ['美团', '携程']:
                        if sheet not in all_sheets:
                            error_messages.append(f"文件 {file_name} 中不存在 '{sheet}' 工作表，已跳过该表")
                            continue
                        
                        # 读取工作表数据
                        if sheet == '美团':
                            df_sheet = pd.read_excel(excel_file, converters={'美团订单号': str}, sheet_name=sheet)
                            if not df_sheet.empty:
                                df_sheet = df_sheet.reset_index(drop=True)  # 重置索引
                                df_sheet = df_sheet[['美团订单号', '分成金额']]
                                df_sheet.rename(columns={'美团订单号': '订单号'}, inplace=True)
                                dfs.append(df_sheet)
                        if sheet == '携程':
                            df_sheet = pd.read_excel(excel_file, converters={'订单号': str}, sheet_name=sheet)
                            if not df_sheet.empty:
                                df_sheet = df_sheet.reset_index(drop=True)  # 重置索引
                                df_sheet = df_sheet[['订单号', '分成金额']]
                                dfs.append(df_sheet)
                except Exception as e:
                    error_messages.append(f"处理文件 {file_name} 时出错：{str(e)}")
            st.success(f"已成功处理 {len(yn_upload_files_ota)} 个文件")
            
            # 3. 显示错误信息（如果有）
            if error_messages:
                st.error("处理过程中出现以下问题：")
                for msg in error_messages:
                    st.error(f"- {msg}")
            
            # 4. 合并数据并存储
            if dfs:  # 只有当有有效数据时才合并
                combined_df = pd.concat(dfs, ignore_index=True)  # 合并所有工作表数据
                st.session_state.yn_ota_df = combined_df  # 直接赋值DataFrame（无需再用read_excel）
                
                st.success(f"所有文件处理完成，共合并 {len(combined_df)} 行数据")
                st.subheader("合并后的数据预览（前3行）")
                st.dataframe(combined_df.head(3), use_container_width=True)
            else:
                st.warning("未读取到任何有效数据，请检查文件内容")
    
    # 对账按钮
    if st.button('开始对账'):
        # 检查文件是否都已上传
        if st.session_state.yn_pms_df is None or st.session_state.yn_ota_df is None:
            st.error('请上传PMS和ota文件')
            return
        
        # 数据处理
        with st.spinner('正在处理数据...'):  
            # 处理PMS数据
            processed_pms = yn_process_pms_data_for_ota(st.session_state.yn_pms_df)
            
            # 处理ota数据
            processed_ota = yn_process_ota_data(st.session_state.yn_ota_df)
            
            # 比对数据
            st.session_state.yn_compare_result = yn_compare_data(processed_pms, processed_ota)
            
            # 重置保存状态
            st.session_state.yn_file_saved = False
            
            st.success('对账完成！')
    
    # 显示对账结果（如果有）
    if st.session_state.yn_compare_result is not None:
        st.subheader('对账结果')
        
        # 结果表格
        st.dataframe(
            st.session_state.yn_compare_result[['核对单号', '消费', '订单号', '分成金额', '差异']],
            use_container_width=True
        )
        
        # 差异分析
        total_days = len(st.session_state.yn_compare_result)
        matched_days = len(st.session_state.yn_compare_result[st.session_state.yn_compare_result['差异'] == 0])
        unmatched_days = total_days - matched_days
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总比对天数", total_days)
        col2.metric("匹配天数", matched_days)
        col3.metric("未匹配天数", unmatched_days)
        
        # # 差异图表
        # if unmatched_days > 0:
        #     st.subheader('差异分析')
        #     fig = st.session_state.yn_compare_result[st.session_state.yn_compare_result['差异'] != 0].plot(
        #         x='核对单号', 
        #         y='差异', 
        #         kind='bar', 
        #         title='订单差异金额',
        #         figsize=(10, 6)
        #     )
        #     st.pyplot(fig.figure)
        
        # 保存和下载按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('保存对账结果'):
                if st.session_state.yn_compare_result is not None:
                    # 生成文件名并保存文件
                    st.session_state.yn_file_name = f'雅南PMS_OTA对账结果_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx'
                    st.session_state.yn_compare_result.to_excel(st.session_state.yn_file_name, index=False)
                    st.session_state.yn_file_saved = True
                    st.success('对账结果已保存')
        
        with col2:
            if st.session_state.yn_file_saved and st.session_state.yn_file_name:
                # 提供下载链接
                with open(st.session_state.yn_file_name, 'rb') as f:
                    st.download_button(
                        label="下载雅南对账结果",
                        data=f,
                        file_name=st.session_state.yn_file_name,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                    
def yn_hotel_pms_pos_check():
    st.write('雅南酒店PMS与POS对账系统')
    
    # 初始化会话状态
    if 'yn_pms_df' not in st.session_state:
        st.session_state.yn_pms_df = None
    if 'yn_pos_df' not in st.session_state:
        st.session_state.yn_pos_df = None
    if 'yn_compare_result' not in st.session_state:
        st.session_state.yn_compare_result = None
    if 'yn_file_saved' not in st.session_state:
        st.session_state.yn_file_saved = False
    if 'yn_file_name' not in st.session_state:
        st.session_state.yn_file_name = None
    
    # 文件上传
    col1, col2 = st.columns(2)
    
    with col1:
        yn_upload_file_pms = st.file_uploader("上传PMS文件，请注意将xls文档另存为xlsx", type=["xlsx"], key='yn_pms_pos')
        if yn_upload_file_pms is not None:
            st.session_state.yn_pms_df = pd.read_excel(yn_upload_file_pms, header=3)[:-1]
            st.success('PMS文件已成功加载')
            st.write('PMS文件前3行预览:')
            st.dataframe(st.session_state.yn_pms_df.head(3))
    
    with col2:
        yn_upload_file_pos = st.file_uploader("上传POS文件", type=["csv"])
        if yn_upload_file_pos is not None:
            # encoding = detect_encoding(upload_file_pos)
            # st.write(encoding)
            st.session_state.yn_pos_df = pd.read_csv(yn_upload_file_pos, header=3, encoding='gbk')
            st.success('POS文件已成功加载')
            st.write('POS文件前3行预览:')
            st.dataframe(st.session_state.yn_pos_df.head(3))
    
    # 对账按钮
    if st.button('开始对账'):
        # 检查文件是否都已上传
        if st.session_state.yn_pms_df is None or st.session_state.yn_pos_df is None:
            st.error('请上传PMS和POS两个文件')
            return
        
        # 数据处理
        with st.spinner('正在处理数据...'):
            pms_calc_list = ['银联', 'POS机微信', 'POS机支付宝']
            
            # 处理PMS数据
            processed_pms = gm_fg_yn_process_pms_data(st.session_state.yn_pms_df, pms_calc_list)

            processed_pos = gm_yn_process_pos_data(st.session_state.yn_pos_df)
            
            # 比对数据
            st.session_state.yn_compare_result = gm_fg_yn_compare_data(processed_pms, processed_pos)

            
            # 重置保存状态
            st.session_state.yn_file_saved = False

            
            st.success('对账完成！')
    
    # 显示对账结果（如果有）
    if st.session_state.yn_compare_result is not None:

        st.subheader('对账结果')
        
        # 结果表格
        st.dataframe(
            st.session_state.yn_compare_result[['日期', '核算金额', '表二时段交易总和', '差异']],

            use_container_width=True
        )
        
        total_days = len(st.session_state.yn_compare_result)
        matched_days = len(st.session_state.yn_compare_result[st.session_state.yn_compare_result['差异'] == 0])
        matched_days = len(st.session_state.yn_compare_result[st.session_state.yn_compare_result['差异'] == 0])
        unmatched_days = total_days - matched_days
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总比对天数", total_days)
        col2.metric("匹配天数", matched_days)
        col3.metric("未匹配天数", unmatched_days)
        
        # # 差异图表
        # if unmatched_days > 0:
        #     st.subheader('差异分析')
        #     fig = st.session_state.gm_compare_result[st.session_state.gm_compare_result['差异'] != 0].plot(
        #         x='日期', 
        #         y='差异', 
        #         kind='bar', 
        #         title='每日差异金额',
        #         figsize=(10, 6)
        #     )
        #     st.pyplot(fig.figure)
        
        # 保存和下载按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('保存对账结果'):
                if st.session_state.yn_compare_result is not None:
                    # 生成文件名并保存文件
                    st.session_state.yn_file_name = f'雅南PMS_POS对账结果_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx'
                    st.session_state.yn_compare_result.to_excel(st.session_state.yn_file_name, index=False)
                    st.session_state.yn_file_saved = True
                    st.success('对账结果已保存')
        
        with col2:
            if st.session_state.yn_file_saved and st.session_state.yn_file_name:
                # 提供下载链接
                with open(st.session_state.yn_file_name, 'rb') as f:
                    st.download_button(
                        label="下载雅南对账结果",
                        data=f,
                        file_name=st.session_state.yn_file_name,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

def lz_hotel_pms_pos_check():
    st.write('兰兹酒店PMS与POS对账系统')
    
    # 初始化会话状态
    if 'lz_pms_df' not in st.session_state:
        st.session_state.lz_pms_df = None
    if 'lz_pos_df' not in st.session_state:
        st.session_state.lz_pos_df = None
    if 'lz_compare_result' not in st.session_state:
        st.session_state.lz_compare_result = None
    if 'lz_file_saved' not in st.session_state:
        st.session_state.lz_file_saved = False
    if 'lz_file_name' not in st.session_state:
        st.session_state.lz_file_name = None
    
    # 文件上传
    col1, col2 = st.columns(2)
    
    with col1:
        lz_upload_file_pms = st.file_uploader("上传PMS文件", type=["xlsx"], key='lz_pms')
        if lz_upload_file_pms is not None:
            lz_pms_df = pd.read_excel(lz_upload_file_pms, sheet_name='结算项', header=3)
            st.session_state.lz_pms_df = lz_pms_df.rename(columns={'账务日期': '日期'})
            st.success('PMS文件已成功加载')
            st.write('PMS文件前3行预览:')
            st.dataframe(st.session_state.lz_pms_df.head(3))
    
    with col2:
        lz_upload_file_pos = st.file_uploader("上传POS文件", type=["xlsx"])
        if lz_upload_file_pos is not None:
            # encoding = detect_encoding(upload_file_pos)
            # st.write(encoding)
            st.session_state.lz_pos_df = pd.read_excel(lz_upload_file_pos)
            st.success('POS文件已成功加载')
            st.write('POS文件前3行预览:')
            st.dataframe(st.session_state.lz_pos_df.head(3))
    
    # 对账按钮
    if st.button('开始对账'):
        # 检查文件是否都已上传
        if st.session_state.lz_pms_df is None or st.session_state.lz_pos_df is None:
            st.error('请上传PMS和POS两个文件')
            return
        
        # 数据处理
        with st.spinner('正在处理数据...'):
            pms_calc_list = ['国内卡', '国外卡', '支付宝', '微信']
            
            # 处理PMS数据
            processed_pms = gm_fg_yn_process_pms_data(st.session_state.lz_pms_df, pms_calc_list)
            processed_pms = processed_pms.groupby('日期')['核算金额'].sum().reset_index()
            
            # 处理POS数据
            processed_pos = fg_lz_process_pos_data(st.session_state.lz_pos_df, default_hour=0)
            
            # 比对数据
            st.session_state.lz_compare_result = gm_fg_yn_compare_data(processed_pms, processed_pos)
            
            # 重置保存状态
            st.session_state.lz_file_saved = False
            
            st.success('对账完成！')
    
    # 显示对账结果（如果有）
    if st.session_state.lz_compare_result is not None:
        st.subheader('对账结果')
        
        # 结果表格
        st.dataframe(
            st.session_state.lz_compare_result[['日期', '核算金额', '表二时段交易总和', '差异']],
            use_container_width=True
        )
        
        # 差异分析
        total_days = len(st.session_state.lz_compare_result)
        matched_days = len(st.session_state.lz_compare_result[st.session_state.lz_compare_result['差异'] == 0])
        unmatched_days = total_days - matched_days
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总比对天数", total_days)
        col2.metric("匹配天数", matched_days)
        col3.metric("未匹配天数", unmatched_days)
        
        # # 差异图表
        # if unmatched_days > 0:
        #     st.subheader('差异分析')
        #     fig = st.session_state.lz_compare_result[st.session_state.lz_compare_result['差异'] != 0].plot(
        #         x='日期', 
        #         y='差异', 
        #         kind='bar', 
        #         title='每日差异金额',
        #         figsize=(10, 6)
        #     )
        #     st.pyplot(fig.figure)
        
        # 保存和下载按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('保存对账结果'):
                if st.session_state.lz_compare_result is not None:
                    # 生成文件名并保存文件
                    st.session_state.lz_file_name = f'兰兹PMS_POS对账结果_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx'
                    st.session_state.lz_compare_result.to_excel(st.session_state.lz_file_name, index=False)
                    st.session_state.lz_file_saved = True
                    st.success('对账结果已保存')
        
        with col2:
            if st.session_state.lz_file_saved and st.session_state.lz_file_name:
                # 提供下载链接
                with open(st.session_state.lz_file_name, 'rb') as f:
                    st.download_button(
                        label="下载兰兹对账结果",
                        data=f,
                        file_name=st.session_state.lz_file_name,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )        

def lz_hotel_pms_ota_check():
    st.write('雅南酒店PMS与OTA对账系统')
    
    # 初始化会话状态
    if 'lz_pms_df' not in st.session_state:
        st.session_state.lz_pms_df = None
    if 'lz_ota_df' not in st.session_state:
        st.session_state.lz_ota_df = None
    
    if 'lz_compare_result' not in st.session_state:
        st.session_state.lz_compare_result = None
    if 'lz_file_saved' not in st.session_state:
        st.session_state.lz_file_saved = False
    if 'lz_file_name' not in st.session_state:
        st.session_state.lz_file_name = None
    
    # 文件上传
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lz_upload_file_pms = st.file_uploader("上传PMS文件，请注意将xls文档另存为xlsx", type=["xlsx"], key='lz_pms_ota')
        if lz_upload_file_pms is not None:
            st.session_state.lz_pms_df = pd.read_excel(lz_upload_file_pms, header=3)
            st.success('PMS文件已成功加载')
            st.write('PMS文件前3行预览:')
            st.dataframe(st.session_state.lz_pms_df.head(3))
    
    with col2:
        lz_upload_files_ota = st.file_uploader(
            label="上传ota文件",
            type=["xls"],
            accept_multiple_files=True  # 允许多文件上传
        )
        
        if lz_upload_files_ota:  # 当有文件上传时
            dfs = []  # 存储所有工作表的数据
            error_messages = []  # 存储错误信息
            st.write(f"正在处理 {len(lz_upload_files_ota)} 个文件")
            for file in lz_upload_files_ota:
                file_name = file.name
                try:
                    # 1. 用ExcelFile打开文件（高效读取多工作表）
                    excel_file = pd.ExcelFile(file, engine='xlrd')  # .xlsx需用openpyxl引擎，xls需用xlrd引擎
                    all_sheets = excel_file.sheet_names  # 获取所有工作表名称
                    
                    # 2. 检查并读取"预付订单明细"工作表
                    for sheet in ['预付订单明细']:
                        if sheet not in all_sheets:
                            error_messages.append(f"文件 {file_name} 中不存在 '{sheet}' 工作表，已跳过该表")
                            continue
                        df_sheet = pd.read_excel(excel_file, sheet_name=sheet, dtype={'订单号': str}, header=1)
                        if not df_sheet.empty:
                            df_sheet = df_sheet.reset_index(drop=True)  # 重置索引
                            df_sheet = df_sheet[['订单号', '结算价']]
                            dfs.append(df_sheet)
                        
                except Exception as e:
                    error_messages.append(f"处理文件 {file_name} 时出错：{str(e)}")
            st.success(f"已成功处理 {len(lz_upload_files_ota)} 个文件")
            # 3. 显示错误信息（如果有）
            if error_messages:
                st.error("处理过程中出现以下问题：")
                for msg in error_messages:
                    st.error(f"- {msg}")
            # 4. 合并数据并存储
            if dfs:  # 只有当有有效数据时才合并
                combined_df = pd.concat(dfs, ignore_index=True)  # 合并所有工作表数据
                st.session_state.lz_ota_df = combined_df  # 直接赋值DataFrame（无需再用read_excel）
                
                st.success(f"所有文件处理完成，共合并 {len(combined_df)} 行数据")
                st.subheader("合并后的数据预览（前3行）")
                st.dataframe(combined_df.head(3), use_container_width=True)
            else:
                st.warning("未读取到任何有效数据，请检查文件内容")
        with col3:
            lz_order_convert_file = st.file_uploader("上传历史房单转换文件", type=["xlsx"])
            if lz_order_convert_file is not None:
                # 读取历史房单转换文件
                lz_order_convert_df = pd.read_excel(lz_order_convert_file, header=3)
                st.session_state.lz_order_convert_df = lz_order_convert_df
                st.success('历史房单文件已成功加载')
                st.write('历史房单文件前3行预览:')
                st.dataframe(st.session_state.lz_order_convert_df.head(3))
                
    # 对账按钮'同单号消费总和'
    if st.button('开始对账'):
        # 检查文件是否都已上传
        if st.session_state.lz_pms_df is None or st.session_state.lz_ota_df is None:
            st.error('请上传PMS和ota文件')
            return
        
        # 数据处理
        with st.spinner('正在处理数据...'):  
            # 处理PMS数据
            processed_pms = lz_process_pms_data_for_ota(st.session_state.lz_pms_df, st.session_state.lz_order_convert_df)
            
            # 处理ota数据
            processed_ota = lz_process_ota_data(st.session_state.lz_ota_df)
            
            # 比对数据
            st.session_state.lz_compare_result = lz_compare_data(processed_pms, processed_ota)
            
            # 重置保存状态
            st.session_state.lz_file_saved = False
            
            st.success('对账完成！')
    
    # 显示对账结果（如果有）
    if st.session_state.lz_compare_result is not None:
        st.subheader('对账结果')
        
        # 结果表格 compare_result.rename(columns={'单号': 'PMS_内部单号', 'OTA单号': 'PMS_OTA单号', '订单号': 'OTA单号'}, inplace=True)
        st.dataframe(
            st.session_state.lz_compare_result[['PMS_内部单号', 'PMS_OTA单号', 'OTA单号', '最早入住时间', '最晚退房时间', 'PMS同单号消费总和', 'OTA同订单号结算价总和', '差异']],
            use_container_width=True
        )
        
        # 差异分析
        total_orders = len(st.session_state.lz_compare_result)
        matched_orders = len(st.session_state.lz_compare_result[st.session_state.lz_compare_result['差异'] == 0])
        unmatched_orders = total_orders - matched_orders
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总比对订单数", total_orders)
        col2.metric("匹配订单数", matched_orders)
        col3.metric("未匹配订单数", unmatched_orders)

        
        # 保存和下载按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('保存对账结果'):
                if st.session_state.lz_compare_result is not None:
                    # 生成文件名并保存文件
                    st.session_state.lz_file_name = f'雅南PMS_OTA对账结果_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx'
                    st.session_state.lz_compare_result.to_excel(st.session_state.lz_file_name, index=False)
                    st.session_state.lz_file_saved = True
                    st.success('对账结果已保存')
        
        with col2:
            if st.session_state.lz_file_saved and st.session_state.lz_file_name:
                # 提供下载链接
                with open(st.session_state.lz_file_name, 'rb') as f:
                    st.download_button(
                        label="下载雅南对账结果",
                        data=f,
                        file_name=st.session_state.lz_file_name,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                   )


# 侧边栏导航
with st.sidebar:
    selected_tab = st.radio("选择功能", ["格曼POS", "方格POS", "雅南OTA", "雅南POS", "兰兹POS", "兰兹OTA"], on_change=reset_state)

# 主界面
if selected_tab == "格曼POS":
    st.title("本程序用于格曼对账")
    st.divider()
    gm_hotel_pms_pos_check()
    st.divider()
    st.caption("© 2025 格曼酒店对账系统 | 版本 1.0")
if selected_tab == "方格POS":
    st.title("本程序用于方格对账")
    st.divider()
    fg_hotel_pms_pos_check()
    st.divider()
    st.caption("© 2025 方格酒店对账系统 | 版本 1.0")
if selected_tab == "雅南OTA":
    st.title("本程序用于雅南对账")
    st.divider()
    yn_hotel_pms_ota_check()
    st.divider()
    st.caption("© 2025 雅南酒店对账系统 | 版本 1.0")
if selected_tab == "雅南POS":
    st.title("本程序用于雅南对账")
    st.divider()
    yn_hotel_pms_pos_check()
    st.divider()
    st.caption("© 2025 雅南酒店对账系统 | 版本 1.0")
if selected_tab == "兰兹POS":
    st.title("本程序用于兰兹对账")
    st.divider()
    lz_hotel_pms_pos_check()
    st.divider()
    st.caption("© 2025 兰兹酒店对账系统 | 版本 1.0")
if selected_tab == "兰兹OTA":
    st.title("本程序用于兰兹对账")
    st.divider()
    lz_hotel_pms_ota_check()
    st.divider()
    st.caption("© 2025 兰兹酒店对账系统 | 版本 1.0")
