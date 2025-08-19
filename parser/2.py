import requests
from bs4 import BeautifulSoup
from datetime import datetime

title = "Агузия"
api_url = "https://warhammer40k.fandom.com/ru/api.php"

# 1. Получаем URL статьи
query_params = {
    "action": "query",
    "titles": title,
    "format": "json",
    "prop": "info",
    "inprop": "url",
}
resp = requests.get(api_url, params=query_params)
resp.raise_for_status()
query_data = resp.json()
pages = query_data["query"]["pages"]
page_info = next(iter(pages.values()))
page_url = page_info.get("fullurl", "URL не найден")

# 2. Получаем содержимое статьи
parse_params = {
    "action": "parse",
    "page": title,
    "format": "json",
    "prop": "text",
    "disabletoc": 1,
    "redirects": True,
}
resp = requests.get(api_url, params=parse_params)
resp.raise_for_status()
parse_data = resp.json()
html = parse_data["parse"]["text"]["*"]
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(separator="\n", strip=True)

print("URL статьи:", page_url)
print("\nТекст статьи:")
print(text[:500], "...")  # выводим первые 500 символов

# 3. Получаем просмотры статьи за последние ~60 дней
views_params = {
    "action": "query",
    "titles": title,
    "format": "json",
    "prop": "pageviews"
}
resp = requests.get(api_url, params=views_params)
resp.raise_for_status()
views_data = resp.json()
pages = views_data["query"]["pages"]
page_info = next(iter(pages.values()))
pageviews = page_info.get("pageviews", {})

# Считаем статистику за последние 30 дней
views_last_30 = {day: count for day, count in list(pageviews.items())[:30] if count is not None}
total_views = sum(views_last_30.values())
avg_views = total_views / len(views_last_30) if views_last_30 else 0

print("\nПопулярность статьи:")
print("Всего просмотров за 30 дней:", total_views)
print("Среднее в день:", round(avg_views, 2))
