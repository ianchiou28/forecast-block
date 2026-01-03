"""
A股板块涨停预测系统 - 定时调度器
支持每日定时执行预测任务
"""
import schedule
import time
import threading
from datetime import datetime
import logging
from typing import Callable, Optional

from config.settings import DATA_CONFIG

logger = logging.getLogger(__name__)


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self):
        self.running = False
        self._thread = None
    
    def add_daily_task(self, time_str: str, task_func: Callable, task_name: str = ""):
        """
        添加每日定时任务
        
        Args:
            time_str: 执行时间，格式 "HH:MM"
            task_func: 任务函数
            task_name: 任务名称
        """
        schedule.every().day.at(time_str).do(
            self._run_task_wrapper, task_func, task_name
        )
        logger.info(f"添加定时任务: {task_name or task_func.__name__} @ {time_str}")
    
    def _run_task_wrapper(self, task_func: Callable, task_name: str):
        """任务执行包装器"""
        logger.info(f"开始执行任务: {task_name}")
        start_time = datetime.now()
        
        try:
            task_func()
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"任务完成: {task_name}, 耗时: {elapsed:.2f}秒")
        except Exception as e:
            logger.error(f"任务执行失败: {task_name}, 错误: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """启动调度器"""
        if self.running:
            logger.warning("调度器已在运行")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("调度器已启动")
    
    def _run_loop(self):
        """调度循环"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """停止调度器"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("调度器已停止")
    
    def run_now(self, task_func: Callable, task_name: str = ""):
        """立即执行任务"""
        self._run_task_wrapper(task_func, task_name or "立即执行")
    
    def get_next_run_time(self) -> Optional[str]:
        """获取下次执行时间"""
        jobs = schedule.get_jobs()
        if not jobs:
            return None
        
        next_run = min(job.next_run for job in jobs)
        return next_run.strftime("%Y-%m-%d %H:%M:%S")


def is_trading_day() -> bool:
    """
    判断今天是否是交易日
    简单实现：周一到周五（不考虑节假日）
    """
    today = datetime.now()
    # 0=周一, 6=周日
    return today.weekday() < 5


def wait_for_time(target_time: str):
    """
    等待到指定时间
    
    Args:
        target_time: 目标时间，格式 "HH:MM"
    """
    while True:
        now = datetime.now()
        target = datetime.strptime(
            f"{now.strftime('%Y-%m-%d')} {target_time}",
            "%Y-%m-%d %H:%M"
        )
        
        if now >= target:
            break
        
        sleep_seconds = (target - now).total_seconds()
        logger.info(f"等待到 {target_time}, 还需 {sleep_seconds:.0f} 秒")
        
        # 每分钟检查一次
        time.sleep(min(60, sleep_seconds))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试调度器
    scheduler = TaskScheduler()
    
    def test_task():
        print(f"测试任务执行: {datetime.now()}")
    
    # 添加测试任务
    scheduler.add_daily_task("08:00", test_task, "早盘预测")
    scheduler.add_daily_task("15:05", test_task, "数据更新")
    
    print(f"下次执行时间: {scheduler.get_next_run_time()}")
    
    # 立即执行一次
    scheduler.run_now(test_task, "手动测试")
