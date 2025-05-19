#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:02:21 2025

@author: hdragon689
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
import requests
import twstock
import os
import json
import time

# --- Configuration & Constants ---
CRITERIA_KEYS = {
    "pe": "max_pe", "pb": "max_pb", "roe": "min_roe", "eps": "min_eps",
    "div_yield": "min_dividend_yield", "avg_div": "min_avg_dividend_payout_5y",
    "beta": "max_beta"
}
CRITERIA_LABELS_CH = {
    CRITERIA_KEYS["pe"]: "本益比 (P/E) 上限", CRITERIA_KEYS["pb"]: "股價淨值比 (P/B) 上限",
    CRITERIA_KEYS["roe"]: "股東權益報酬率 (ROE) 下限 (%)", CRITERIA_KEYS["eps"]: "每股盈餘 (EPS) 下限",
    CRITERIA_KEYS["div_yield"]: "殖利率 (%) 下限", CRITERIA_KEYS["avg_div"]: "近五年平均現金股利 (元) 下限",
    CRITERIA_KEYS["beta"]: "Beta (β) 上限"
}
COLUMN_NAMES_CH = { # These are the keys used internally by yfinance/script BEFORE localization for display
    "Ticker": "股票代號", "Name": "公司名稱", "Price": "目前股價", "P/E": "本益比",
    "P/B": "股價淨值比", "ROE": "股東權益報酬率 (%)", "EPS": "每股盈餘 (EPS)",
    "Div Yield (殖利率)": "殖利率 (%)", "Avg Div 5Y (過去五年平均配息)": "近五年平均現金股利 (元)",
    "Beta": "Beta (β)", "Currency": "幣別", "Market Cap": "市值"
}
DEFAULT_CRITERIA_VALUES = {
    CRITERIA_KEYS["pe"]: 20.0, CRITERIA_KEYS["pb"]: 2.5, CRITERIA_KEYS["roe"]: 0.15, # ROE as decimal
    CRITERIA_KEYS["eps"]: 1.0, CRITERIA_KEYS["div_yield"]: 0.03, # Div Yield as decimal
    CRITERIA_KEYS["avg_div"]: 0.5, CRITERIA_KEYS["beta"]: 1.2
}

CACHE_DIR = "tempdata"
CACHE_TTL_SECONDS = 12 * 60 * 60
os.makedirs(CACHE_DIR, exist_ok=True)

def save_to_file_cache(filename, data):
    filepath = os.path.join(CACHE_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.warning(f"無法儲存快取檔案 {filename}: {e}")

def load_from_file_cache(filename):
    filepath = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(filepath):
        return None
    try:
        file_mod_time = os.path.getmtime(filepath)
        if (time.time() - file_mod_time) > CACHE_TTL_SECONDS:
            os.remove(filepath)
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.warning(f"無法讀取快取檔案 {filename}: {e}")
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except Exception: pass
        return None

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_sp500_tickers_cached_logic():
    filename = "sp500_tickers.json"
    cached_data = load_from_file_cache(filename)
    if cached_data: return cached_data[0], cached_data[1]
    st.write("正在獲取 S&P 500 股票代號 (將存入本地快取12小時)...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; StockScreenerApp/1.0; +http://localhost)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        payload = pd.read_html(response.text)
        tickers = payload[0]['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        names_map = {ticker: "" for ticker in tickers}
        save_to_file_cache(filename, (tickers, names_map))
        return tickers, names_map
    except Exception as e:
        st.error(f"獲取 S&P 500 股票代號失敗: {e}")
        return [], {}

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_nasdaq100_tickers_cached_logic():
    filename = "nasdaq100_tickers.json"
    cached_data = load_from_file_cache(filename)
    if cached_data: return cached_data[0], cached_data[1]
    st.write("正在獲取 NASDAQ 100 股票代號 (將存入本地快取12小時)...")
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; StockScreenerApp/1.0; +http://localhost)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        payload = pd.read_html(response.text)
        nasdaq_df = next((df_table for df_table in payload if 'Ticker' in df_table.columns), None)
        if nasdaq_df is not None:
            tickers = nasdaq_df['Ticker'].tolist()
            names_map = {ticker: "" for ticker in tickers}
            save_to_file_cache(filename, (tickers, names_map))
            return tickers, names_map
        st.error("在維基百科頁面找不到 NASDAQ 100 股票代號表。")
        return [], {}
    except Exception as e:
        st.error(f"獲取 NASDAQ 100 股票代號失敗: {e}")
        return [], {}

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_tw_market_tickers_cached_logic(market_type_filter):
    filename = f"tw_{market_type_filter}_tickers.json"
    cached_data = load_from_file_cache(filename)
    if cached_data: return cached_data[0], cached_data[1]
    display_market_type = "上市 (TWSE)" if market_type_filter == "上市" else "上櫃 (TPEx)"
    st.write(f"正在獲取台灣 {display_market_type} 公司股票代號及名稱 (將存入本地快取12小時, 首次可能較久)...")
    try:
        all_stocks = twstock.codes
        if not all_stocks:
            st.warning(f"`twstock.codes` 未返回任何資料 ({display_market_type})。")
            return [], {}
        tw_tickers, tw_names_map = [], {}
        for code, stock_info in all_stocks.items():
            if stock_info and hasattr(stock_info, 'type') and stock_info.type == '股票' and \
               hasattr(stock_info, 'market') and stock_info.market == market_type_filter and \
               hasattr(stock_info, 'name'):
                ticker_yf = f"{code}.TW"
                tw_tickers.append(ticker_yf)
                tw_names_map[ticker_yf] = stock_info.name
        save_to_file_cache(filename, (tw_tickers, tw_names_map))
        return tw_tickers, tw_names_map
    except Exception as e:
        st.error(f"使用 twstock 獲取台灣 {display_market_type} 股票代號失敗: {e}")
        return [], {}

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_5yr_avg_dividend_from_cached_data(ticker_symbol_for_cache, dividends_data_for_calc):
    if not dividends_data_for_calc:
        return 0.0
    try:
        if isinstance(dividends_data_for_calc, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in dividends_data_for_calc):
            dividends_dict = {pd.to_datetime(item[0]): item[1] for item in dividends_data_for_calc if item[0] is not None and pd.notna(item[1])}
            if not dividends_dict: dividends = pd.Series(dtype='float64')
            else: dividends = pd.Series(dividends_dict)
        else:
            return 0.0

        if dividends.empty: return 0.0
        end_date_local = datetime.now() 
        start_date_local_naive = end_date_local - timedelta(days=5*365 + 180)
        if dividends.index.tz is not None:
            dividends.index = dividends.index.tz_localize(None)
        start_date_to_compare = pd.Timestamp(start_date_local_naive.year, start_date_local_naive.month, start_date_local_naive.day)
        recent_dividends = dividends[dividends.index >= start_date_to_compare]
        if recent_dividends.empty: return 0.0
        annual_dividends = recent_dividends.groupby(recent_dividends.index.year).sum()
        if annual_dividends.empty: return 0.0
        avg_payout = annual_dividends.tail(5).mean()
        return avg_payout if pd.notna(avg_payout) else 0.0
    except Exception as e:
        return 0.0

@st.cache_data(ttl=CACHE_TTL_SECONDS) 
def get_stock_info_file_cached_logic(ticker_symbol, chinese_name_map): # chinese_name_map is a dict
    safe_ticker_filename = ticker_symbol.replace('.', '_').replace(':', '_')
    filename = f"stockinfo_{safe_ticker_filename}.json"
    
    file_cached_stock_data = load_from_file_cache(filename)
    if file_cached_stock_data is not None:
        return file_cached_stock_data

    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info 
        dividends_for_avg_calc = []
        if hasattr(stock, 'dividends') and not stock.dividends.empty:
            dividends_for_avg_calc = [(str(ts.date()), val) for ts, val in stock.dividends.items() if pd.notna(val)]
        if not info or 'symbol' not in info: return None
        avg_div_5y = get_5yr_avg_dividend_from_cached_data(ticker_symbol, dividends_for_avg_calc)
        name_to_use = chinese_name_map.get(ticker_symbol, info.get('shortName') or info.get('longName', ticker_symbol))
        data_to_save = {
            "Ticker": ticker_symbol, "Name": name_to_use,
            "Price": info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose'),
            "P/E": info.get('trailingPE'), "P/B": info.get('priceToBook'),
            "ROE": info.get('returnOnEquity'), "EPS": info.get('trailingEps'),
            "Div Yield (殖利率)": info.get('dividendYield'), # Store as decimal
            "Avg Div 5Y (過去五年平均配息)": avg_div_5y, 
            "Beta": info.get('beta'), "Currency": info.get("currency", "N/A"),
            "Market Cap": info.get("marketCap"),
        }
        save_to_file_cache(filename, data_to_save)
        return data_to_save
    except Exception as e:
        return None

# --- Screen Stocks Function (CORRECTED DataFrame Formatting) ---
def screen_stocks(tickers, criteria_values, active_criteria, progress_bar_st_element, chinese_name_map):
    potential_stocks_data = []
    all_stocks_data = []
    total_tickers = len(tickers)
    processed_count = 0

    for ticker_symbol in tickers:
        processed_count += 1
        if progress_bar_st_element:
            progress_text = f"處理中: {ticker_symbol} ({processed_count}/{total_tickers})"
            progress_bar_st_element.progress(processed_count / total_tickers, text=progress_text)
        
        stock_data = get_stock_info_file_cached_logic(ticker_symbol, chinese_name_map) 
        
        if stock_data:
            all_stocks_data.append(stock_data)
            current_stock_passes = True 
            if active_criteria[CRITERIA_KEYS["pe"]]:
                pe_value = stock_data.get("P/E") 
                if pe_value is None: current_stock_passes = False
                elif not (0 < pe_value <= criteria_values[CRITERIA_KEYS["pe"]]): current_stock_passes = False
            if current_stock_passes and active_criteria[CRITERIA_KEYS["pb"]]:
                pb_value = stock_data.get("P/B")
                if pb_value is None: current_stock_passes = False
                elif not (0 < pb_value <= criteria_values[CRITERIA_KEYS["pb"]]): current_stock_passes = False
            if current_stock_passes and active_criteria[CRITERIA_KEYS["roe"]]:
                roe_value = stock_data.get("ROE") 
                if roe_value is None: current_stock_passes = False
                elif roe_value < criteria_values[CRITERIA_KEYS["roe"]]: current_stock_passes = False
            if current_stock_passes and active_criteria[CRITERIA_KEYS["eps"]]:
                eps_value = stock_data.get("EPS")
                if eps_value is None: current_stock_passes = False
                elif eps_value < criteria_values[CRITERIA_KEYS["eps"]]: current_stock_passes = False
            if current_stock_passes and active_criteria[CRITERIA_KEYS["div_yield"]]:
                div_yield_value = stock_data.get("Div Yield (殖利率)") 
                if div_yield_value is None: current_stock_passes = False
                elif div_yield_value < criteria_values[CRITERIA_KEYS["div_yield"]]: current_stock_passes = False
            if current_stock_passes and active_criteria[CRITERIA_KEYS["avg_div"]]:
                avg_div_value = stock_data.get("Avg Div 5Y (過去五年平均配息)")
                if avg_div_value is None: current_stock_passes = False
                elif avg_div_value < criteria_values[CRITERIA_KEYS["avg_div"]]: current_stock_passes = False
            if current_stock_passes and active_criteria[CRITERIA_KEYS["beta"]]:
                beta_value = stock_data.get("Beta")
                if beta_value is None: pass 
                elif beta_value > criteria_values[CRITERIA_KEYS["beta"]]: current_stock_passes = False
            if current_stock_passes:
                potential_stocks_data.append(stock_data)
                
    if progress_bar_st_element:
        progress_bar_st_element.empty()

    df_potential_formatted = pd.DataFrame(potential_stocks_data)
    df_all_formatted = pd.DataFrame(all_stocks_data)
    
    cols_to_format_numeric_standard = ["Price", "P/E", "P/B", "EPS", "Avg Div 5Y (過去五年平均配息)", "Beta"]

    final_dfs = []
    for df_orig in [df_potential_formatted, df_all_formatted]:
        if not df_orig.empty:
            df = df_orig.copy()

            # Convert to numeric first for all relevant columns
            for col in cols_to_format_numeric_standard + ["ROE", "Div Yield (殖利率)"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Apply rounding to non-percentage numeric columns
            for col in cols_to_format_numeric_standard:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].round(2) # Standard 2 decimal places
            
            # Convert ROE and Dividend Yield to percentage values *in their original columns*
            if "ROE" in df.columns and pd.api.types.is_numeric_dtype(df["ROE"]):
                df["ROE"] = (df["ROE"] * 100).round(2)
            
            if "Div Yield (殖利率)" in df.columns and pd.api.types.is_numeric_dtype(df["Div Yield (殖利率)"]):
                df["Div Yield (殖利率)"] = (df["Div Yield (殖利率)"] * 100).round(2)

            if "Market Cap" in df.columns and pd.api.types.is_numeric_dtype(df["Market Cap"]):
                df["Market Cap"] = df["Market Cap"].apply(
                    lambda x: f"{x/1e12:.2f}兆" if pd.notna(x) and x >= 1e12 else (
                              f"{x/1e8:.2f}億" if pd.notna(x) and x >= 1e8 else (
                              f"{x/1e4:.2f}萬" if pd.notna(x) and x >= 1e4 else (
                              f"{x:.0f}" if pd.notna(x) else "N/A"))))
            
            df.rename(columns=COLUMN_NAMES_CH, inplace=True)
            
            ordered_cols_display_names = [
                COLUMN_NAMES_CH.get("Ticker"), COLUMN_NAMES_CH.get("Name"), 
                COLUMN_NAMES_CH.get("Price"), COLUMN_NAMES_CH.get("P/E"), 
                COLUMN_NAMES_CH.get("P/B"), COLUMN_NAMES_CH.get("ROE"), 
                COLUMN_NAMES_CH.get("EPS"), COLUMN_NAMES_CH.get("Div Yield (殖利率)"),
                COLUMN_NAMES_CH.get("Avg Div 5Y (過去五年平均配息)"), 
                COLUMN_NAMES_CH.get("Beta"), COLUMN_NAMES_CH.get("Market Cap"), 
                COLUMN_NAMES_CH.get("Currency")
            ]
            existing_ordered_cols = [col for col in ordered_cols_display_names if col and col in df.columns]
            
            df = df[existing_ordered_cols]
            final_dfs.append(df)
        else:
            final_dfs.append(pd.DataFrame())
            
    return final_dfs[0], final_dfs[1]

# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="價值股票篩選器", layout="wide")
    st.title("📈 價值股票篩選器")
    st.markdown("""
    從 S&P 500、NASDAQ 100、台灣上市(TWSE)、台灣上櫃(TPEx)公司或自訂列表篩選股票。
    可勾選欲啟用之篩選條件。資料將優先從本地 `tempdata` 資料夾快取 (12小時效期)，其次為記憶體快取。
    """)
    st.sidebar.header("⚙️ 篩選設定")

    market_options_display = {
        "S&P 500 (美國)": "sp500", "NASDAQ 100 (美國)": "nasdaq100",
        "台灣上市 (TWSE)": "tw_listed", "台灣上櫃 (TPEx)": "tw_otc",
        "自訂/其他台灣股票列表": "custom"
    }
    selected_market_displays = st.sidebar.multiselect(
        "選擇市場/指數:", options=list(market_options_display.keys()), default=[]
    )
    selected_markets_internal = [market_options_display[disp] for disp in selected_market_displays]

    custom_tickers_str = ""
    if "custom" in selected_markets_internal:
        st.sidebar.subheader("自訂/其他台灣股票代號")
        custom_tickers_str = st.sidebar.text_area(
            "輸入股票代號 (逗號分隔)", "2330.TW, 2317.TW", height=100
        )

    st.sidebar.subheader("財務篩選條件 (勾選以啟用)")
    criteria_values = {}
    active_criteria = {}

    for key_internal_map, label_ch in CRITERIA_LABELS_CH.items():
        active_criteria[key_internal_map] = st.sidebar.checkbox(f"啟用 {label_ch}", value=True, key=f"cb_{key_internal_map}")
        
        default_val = DEFAULT_CRITERIA_VALUES[key_internal_map]
        min_val_ui = -100.0 if key_internal_map == CRITERIA_KEYS["eps"] else 0.0
        if key_internal_map == CRITERIA_KEYS["pe"] or key_internal_map == CRITERIA_KEYS["pb"]:
            min_val_ui = 0.1
        
        step_val = 0.1
        format_str = "%.1f"
        if key_internal_map == CRITERIA_KEYS["eps"] or key_internal_map == CRITERIA_KEYS["avg_div"]:
            format_str = "%.2f"

        current_val_input = default_val
        label_ui = label_ch
        if key_internal_map == CRITERIA_KEYS["roe"] or key_internal_map == CRITERIA_KEYS["div_yield"]:
            current_val_input = default_val * 100 
            label_ui = label_ch.replace('(%)', '(輸入%值, 如15)')


        user_input = st.sidebar.number_input(
            label_ui, min_value=min_val_ui, value=current_val_input, 
            step=step_val, format=format_str, disabled=not active_criteria[key_internal_map],
            key=f"num_input_{key_internal_map}"
        )
        
        if key_internal_map == CRITERIA_KEYS["roe"] or key_internal_map == CRITERIA_KEYS["div_yield"]:
            criteria_values[key_internal_map] = user_input / 100 
        else:
            criteria_values[key_internal_map] = user_input


    if st.sidebar.button("🚀 開始篩選", type="primary"):
        final_tickers_list, master_chinese_name_map = [], {}
        with st.spinner("正在準備股票代號列表... (檢查本地與記憶體快取)"):
            if "sp500" in selected_markets_internal:
                tickers, names = get_sp500_tickers_cached_logic()
                final_tickers_list.extend(tickers); master_chinese_name_map.update(names)
            if "nasdaq100" in selected_markets_internal:
                tickers, names = get_nasdaq100_tickers_cached_logic()
                final_tickers_list.extend(tickers); master_chinese_name_map.update(names)
            if "tw_listed" in selected_markets_internal:
                tickers, names = get_tw_market_tickers_cached_logic(market_type_filter='上市')
                final_tickers_list.extend(tickers); master_chinese_name_map.update(names)
            if "tw_otc" in selected_markets_internal:
                tickers, names = get_tw_market_tickers_cached_logic(market_type_filter='上櫃')
                final_tickers_list.extend(tickers); master_chinese_name_map.update(names)
            if "custom" in selected_markets_internal and custom_tickers_str:
                custom_list = [ticker.strip().upper() for ticker in custom_tickers_str.split(',') if ticker.strip()]
                final_tickers_list.extend(custom_list)
        
        stock_tickers_to_screen = sorted(list(set(final_tickers_list)))
        if not stock_tickers_to_screen:
            st.warning("請至少選擇一個市場或提供自訂股票代號。")
            return

        st.info(f"開始篩選 {len(stock_tickers_to_screen)} 個不重複的股票代號...")
        progress_bar_placeholder = st.empty()
        progress_bar_st_element = progress_bar_placeholder.progress(0, text="初始化...")

        try:
            qualified_df, all_df = screen_stocks(
                stock_tickers_to_screen, criteria_values, active_criteria,
                progress_bar_st_element, master_chinese_name_map
            )
            
            st.subheader(f"📊 所有已獲取資料的股票 ({len(all_df) if not all_df.empty else 0} 檔)")
            if not all_df.empty: st.dataframe(all_df, use_container_width=True, height=600)
            else: st.write("未獲取任何股票資料。")

            st.subheader(f"✅ 符合已啟用篩選條件的潛力股票 ({len(qualified_df) if not qualified_df.empty else 0} 檔)")
            if not qualified_df.empty: st.dataframe(qualified_df, use_container_width=True, height=400)
            else: st.write("沒有股票符合所有已啟用的篩選條件。")
            st.success("篩選完成！")
        except Exception as e:
            st.error(f"篩選過程中發生錯誤: {e}")
            st.exception(e) # Show full traceback for debugging
    else:
        st.info("請在側邊欄選擇市場、設定篩選條件，然後點擊「開始篩選」。")

    st.sidebar.markdown("---")
    st.sidebar.markdown("技術提供：[Streamlit](https://streamlit.io) & [yfinance](https://pypi.org/project/yfinance/) & [twstock](https://github.com/mlouielu/twstock)")
    st.sidebar.markdown("S&P 500/NASDAQ 100來自維基百科。台灣股票列表來自 `twstock`。本地檔案快取12小時。")

if __name__ == "__main__":
    main()