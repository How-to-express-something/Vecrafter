"""
协议层，定义了前后端通信的协议和接口。

前端StreamLit和命令行界面都要实现write和accept方法
上官记得看
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

"""
message协议定义 还未设计

"""

@dataclass
class Message:
    #瞎写的，后续再完善
    content: str
    metadata: dict = field(default_factory=dict)




class ICommunicationProtocol(ABC):
    @abstractmethod
    def write(self, message: Message):
        """将数据发送到前端显示"""
        pass

    @abstractmethod
    def accept(self) -> Optional[Message]:
        """接受前端发送的数据"""
        pass
