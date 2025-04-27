import openai
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


def split_plaintext_into_trunks(
    plaintext: str, max_words: int = 300, overlap: int = 0
) -> list[str]:
    """
    Split a long plaintext into overlapping chunks (trunks), each with at most `max_words` words,
    and `overlap` words overlapping between consecutive trunks.

    Args:
        plaintext (str): The input text.
        max_words (int): Maximum number of words per chunk.
        overlap (int): Number of overlapping words between adjacent chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    assert 0 <= overlap < max_words, "Overlap must be >= 0 and less than max_words"

    words = plaintext.split()
    trunks = []
    step = max_words - overlap

    for i in range(0, len(words), step):
        trunk = " ".join(words[i : i + max_words])
        trunks.append(trunk)
        if i + max_words >= len(words):
            break

    return trunks


class AliyunChat:
    def __init__(
        self,
        model="Llama-4-Scout-17B-16E-Instruct",
        system_prompt="You are a helpful assistant.",
        max_tokens=400,
    ):
        self.client = openai.OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.messages = [{"role": "system", "content": system_prompt}]
        self.model = model
        self.max_tokens = max_tokens

    def startChat(self, messageStr):
        self.messages.append({"role": "user", "content": messageStr})
        response = self._conversation(self.messages)
        # print(self.messages, response)
        assistentResponse = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistentResponse})
        return assistentResponse

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
    def _conversation(self, messages):
        return self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens
        )

    def resetChat(self):
        # 重新开始对话，清除对话历史
        self.messages = []


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    text = "Dame Deirdre Mary Hutton (born 15 March 1949), is a British public servant, termed by The Daily Telegraph as \"Queen of the Quangos\" and \"The great quango hopper\". She was the chair of the UK's Civil Aviation Authority from 2009 to 2020.\n\nA former anti-apartheid demonstrator who was once arrested in South Africa, after a short private sector career working for Anchor housing association (1973–75), she then became a researcher for Glasgow Chamber of Commerce (1975–80), before becoming its chair (1980–82).\n\nHutton has worked for over 10 major non-departmental public bodies in 30 years. Her first appointment was in 1980 to the Arts Council of Scotland.\n\nShe developed her career in championing consumer issues within public sector bodies, particularly in health and food standards and regulation, including: Chair of the Foresight Panel on the Food Chain and Crops for Industry; Chair of the Food Chain Centre; member of the 2001–2 Policy Commission on the Future of Farming and Food (Curry Commission). She chaired the board of Rural Forum Scotland in 1999 when it collapsed due to insolvency. She was, until June 2008, the Vice-Chair of the European Food Safety Authority Management Board. She is Honorary Vice-President of the Institute of Food Science and Technology.\n\nHutton was a non-executive Director of the Scottish Borders Health Board and a member of The King's Fund Organizational Audit Council. She was a member of the Wilson Committee on Complaints in the National Health Service, and of the General Dental Council.\n\nFor five years until 2005, she was Chair of the National Consumer Council, having formerly chaired the Scottish Consumer Council. She was Vice-Chair of the Scottish Environment Protection Agency, a member of the Sustainable Development Commission and a member of the Energy Advisory Panel for the UK Department of Trade and Industry. She was a member of the Better Regulation Task Force. Chair of the Personal Investment Authority Ombudsman Council, Hutton was then Deputy Chair of the Financial Services Authority until December 2007. She was a member of the Secretary of State's Consultative Steering Group on the Scottish Parliament.\n\nDuring 2008, she was on the three-member panel that conducted an independent review of the postal services on behalf of the Department for Business, Enterprise and Regulatory Reform. Hutton was (2011–16) a non-executive Director of Castle Trust, and non-executive member of the Treasury Board, and Thames Water.\n\nHutton is one of 32 Vice-Presidents of the Chartered Trading Standards Institute.\n\nAppointed to the board of the UK Civil Aviation Authority (CAA) as a non-executive director in April 2009, Hutton was appointed chair in 2009 by Transport Secretary Geoff Hoon, replacing Sir Roy McNulty; she was paid £130,000 for two days' work a week in 2010, which was still the case as of 2015, making her one of the 328 most highly paid people in the British public sector at that time. She retired from the role in 2020. On 1 August 2020, she was appointed as Chancellor of Cranfield University.\n\nDeirdre Mary Cassells married Alasdair Henry Hutton in 1975 in Oxford. She was appointed CBE in the Queen's Birthday Honours for 1998, and advanced to DBE in the Birthday List of 2004. In April 2010, she was awarded a Fellowship of City and Guilds.\n\nHutton has two sons, Thomas and Nicholas Hutton. Her hobbies include gardening and chamber music. She divorced from Alasdair Hutton.\n\n\n\n\n * Biodata, Debretts.com\n * Profile, lboro.ac.uk\n * Biodata, food.gov.uk"
    for i, trunk in enumerate(
        split_plaintext_into_trunks(text, max_words=400, overlap=20)
    ):
        print(i, trunk)

    # model="llama-4-scout-17b-16e-instruct" # 10
    # model="llama-4-maverick-17b-128e-instruct" # 10
    # model="deepseek-r1-distill-qwen-1.5b" # 60
    # model="deepseek-r1-distill-llama-8b" # 60
    model = "deepseek-r1-distill-llama-70b"  # 60
    ChatAI = AliyunChat(
        model=model,
    )
    assistantReplay = ChatAI.startChat("hello")
    print(assistantReplay)
