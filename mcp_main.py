from mcp.server import FastMCP

# # 初始化 FastMCP 服务器
app = FastMCP('get_time')

@app.tool()
async def get_time(address: str) -> str:
    """
    获取地区当前时间

    Args:
        address: 查询地区

    Returns:
        搜索结果的总结
    """
    x = {
        "time":address+"2025/05/05/11/07"
    }
    return x.get('time')




if __name__ == '__main__':

    app.run(transport='stdio')


